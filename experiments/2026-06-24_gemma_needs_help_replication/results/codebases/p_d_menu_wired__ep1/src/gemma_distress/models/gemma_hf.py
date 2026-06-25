"""Local Gemma client via HuggingFace transformers.

Handles three needs the API providers cannot:
  * instruct chat with the Gemma chat template;
  * base ("-pt") continuation for the Section 3 prefilling experiment;
  * assistant-turn prefilling for both base and instruct models
    (``continue_final_message`` for instruct; raw concatenation for base);
  * optional LoRA adapter loading to evaluate DPO/SFT-finetuned Gemma.

The model is loaded lazily on first ``generate`` so that constructing the
registry (and importing this module) is cheap and side-effect free.
"""
from __future__ import annotations

from typing import Any, Sequence

from .base import ChatModel, GenResult, Message


class GemmaHFModel(ChatModel):
    def __init__(
        self,
        name: str,
        hf_id: str,
        kind: str = "instruct",
        adapter_path: str | None = None,
        device_map: str = "auto",
        dtype: str = "bfloat16",
    ) -> None:
        super().__init__(name=name, kind=kind)
        self.hf_id = hf_id
        self.adapter_path = adapter_path
        self.device_map = device_map
        self.dtype = dtype
        self._model = None
        self._tokenizer = None

    def _load(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        torch_dtype = getattr(torch, self.dtype, torch.bfloat16)
        self._tokenizer = AutoTokenizer.from_pretrained(self.hf_id)
        self._model = AutoModelForCausalLM.from_pretrained(
            self.hf_id, torch_dtype=torch_dtype, device_map=self.device_map
        )
        if self.adapter_path:
            from peft import PeftModel

            self._model = PeftModel.from_pretrained(self._model, self.adapter_path)
        self._model.eval()

    # -- prompt construction ---------------------------------------------------
    def _build_input_text(
        self,
        messages: Sequence[Message],
        system: str | None,
        prefill: str | None,
    ) -> str:
        tok = self._tokenizer
        msgs = list(messages)
        if system:
            msgs = [{"role": "system", "content": system}] + msgs

        if self.is_base:
            # Base models are not chat-tuned: render a lightweight plain-text
            # transcript and let the model continue. The prefill (if any) seeds
            # the start of the assistant's reply.
            lines = []
            for m in msgs:
                tag = {"system": "System", "user": "User", "assistant": "Assistant"}[
                    m["role"]
                ]
                lines.append(f"{tag}: {m['content']}")
            text = "\n".join(lines) + "\nAssistant:"
            if prefill:
                text += " " + prefill
            return text

        # Instruct: use the official chat template.
        if prefill:
            # Open the assistant turn with the prefill and continue it.
            msgs = msgs + [{"role": "assistant", "content": prefill}]
            return tok.apply_chat_template(
                msgs,
                tokenize=False,
                continue_final_message=True,
            )
        return tok.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True
        )

    def generate(
        self,
        messages: Sequence[Message],
        *,
        temperature: float = 1.0,
        max_tokens: int = 2048,
        prefill: str | None = None,
        stop: Sequence[str] | None = None,
        tools: Sequence[dict[str, Any]] | None = None,
        system: str | None = None,
    ) -> GenResult:
        self._load()
        import torch

        input_text = self._build_input_text(messages, system, prefill)
        inputs = self._tokenizer(input_text, return_tensors="pt").to(
            self._model.device
        )
        prompt_len = inputs["input_ids"].shape[1]

        gen_kwargs: dict[str, Any] = dict(
            max_new_tokens=max_tokens,
            do_sample=temperature > 0,
            temperature=temperature if temperature > 0 else None,
            pad_token_id=self._tokenizer.pad_token_id
            or self._tokenizer.eos_token_id,
        )

        with torch.no_grad():
            out = self._model.generate(**inputs, **gen_kwargs)

        new_tokens = out[0][prompt_len:]
        text = self._tokenizer.decode(new_tokens, skip_special_tokens=True)

        # Software stop-sequence handling (HF generate has no native string stop).
        if stop:
            cut = len(text)
            for s in stop:
                idx = text.find(s)
                if idx != -1:
                    cut = min(cut, idx)
            text = text[:cut]

        # For base models, trim at the next "User:"/"System:" turn boundary.
        if self.is_base:
            for boundary in ("\nUser:", "\nSystem:", "\nAssistant:"):
                idx = text.find(boundary)
                if idx != -1:
                    text = text[:idx]

        return GenResult(text=text.strip(), stop_reason="stop", raw=out)
