"""Local HuggingFace backend for Gemma (instruct + base).

Handles three things the paper relies on:
  * chat-templated generation for the instruct models;
  * **prefilling** an assistant turn (Section 3 "early"/"onset" truncation, and
    making the base model continue a chat-formatted conversation);
  * loading a **LoRA adapter** on top of the base checkpoint (the DPO/SFT
    finetuned Gemma).

Base ("pt") models are not chat-trained; we render the conversation with the
same chat template but always rely on a prefill so the base model continues a
concrete assistant turn rather than being asked to follow chat formatting it
never saw (this is exactly the paper's prefilling methodology, Section 3.1).
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from .base import ChatMessage

if TYPE_CHECKING:
    from ..config import ModelSpec


class HFChatClient:
    def __init__(
        self,
        spec: "ModelSpec",
        *,
        adapter_path: str | None = None,
        dtype: str = "bfloat16",
        device_map: str = "auto",
        load_in_4bit: bool = False,
        max_memory: dict | None = None,
    ):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.spec = spec
        self.adapter_path = adapter_path
        self._torch = torch

        self.tokenizer = AutoTokenizer.from_pretrained(spec.model_id)

        quant_kwargs = {}
        if load_in_4bit:
            from transformers import BitsAndBytesConfig
            quant_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=getattr(torch, dtype),
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )

        self.model = AutoModelForCausalLM.from_pretrained(
            spec.model_id,
            torch_dtype=getattr(torch, dtype),
            device_map=device_map,
            max_memory=max_memory,
            **quant_kwargs,
        )

        if adapter_path:
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, adapter_path)

        self.model.eval()

    # ------------------------------------------------------------------ #
    def _render_prompt(self, messages: list[ChatMessage], prefill: str | None) -> str:
        """Render messages with the chat template, then append the prefill so
        the model continues the assistant turn from `prefill`."""
        msg_dicts = [{"role": m.role, "content": m.content} for m in messages]
        text = self.tokenizer.apply_chat_template(
            msg_dicts,
            tokenize=False,
            add_generation_prompt=True,
        )
        if prefill:
            text = text + prefill
        return text

    def generate(
        self,
        messages: list[ChatMessage],
        *,
        max_new_tokens: int = 1024,
        temperature: float = 1.0,
        prefill: str | None = None,
        n: int = 1,
    ) -> list[str]:
        torch = self._torch
        prompt_text = self._render_prompt(messages, prefill)
        inputs = self.tokenizer(prompt_text, return_tensors="pt",
                                add_special_tokens=False).to(self.model.device)
        prompt_len = inputs["input_ids"].shape[1]

        gen_kwargs = dict(
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0,
            temperature=max(temperature, 1e-6),
            top_p=1.0,
            num_return_sequences=n,
            pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
        )
        with torch.no_grad():
            out = self.model.generate(**inputs, **gen_kwargs)

        # Strip the prompt (and prefill) — return only the newly generated text,
        # matching the paper's "continuation (excluding prefill)".
        completions = []
        for seq in out:
            gen_ids = seq[prompt_len:]
            completions.append(
                self.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
            )
        return completions
