"""Local HuggingFace client for Gemma 3 (instruct, pretrained, and LoRA-adapted).

Handles three cases the experiments need:

1. **Instruct chat** -- apply the Gemma chat template and sample a turn.
2. **Prefill / continuation** -- continue a half-written assistant turn
   (``continue_final_message=True``). Used for Section 3 and the recovery test.
3. **Base (pretrained) models** -- ``google/gemma-3-27b-pt`` has no chat
   template, so we render a plain-text transcript ourselves (see
   ``_render_base_transcript``) and let the model continue it.

A LoRA adapter (the DPO/SFT output of Section 4) can be layered on top via
``adapter_path``.
"""
from __future__ import annotations

import os

import config
from .base import Message, ModelClient

# Plain-text role tags for base-model transcripts. Base models were never
# trained on a chat format; we use a minimal, neutral scaffold and rely on
# prefilling so the comparison to the instruct model is about content, not
# format (see DESIGN.md, "Base-model transcript format").
_BASE_USER_TAG = "User:"
_BASE_ASSISTANT_TAG = "Assistant:"


class HFGemmaClient(ModelClient):
    def __init__(self, spec: "config.ModelSpec", adapter_path: str | None = None,
                 load_in_4bit: bool = False, device_map: str = "auto"):
        super().__init__(spec)
        self.adapter_path = adapter_path
        self._load_in_4bit = load_in_4bit
        self._device_map = device_map
        self._model = None
        self._tokenizer = None

    # ------------------------------------------------------------------ #
    # Lazy load (so importing the module is cheap and GPU-free)
    # ------------------------------------------------------------------ #
    def _ensure_loaded(self):
        if self._model is not None:
            return
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        token = os.environ.get(config.HF_TOKEN_ENV)
        kwargs: dict = {"torch_dtype": torch.bfloat16, "device_map": self._device_map}
        if self._load_in_4bit:
            from transformers import BitsAndBytesConfig
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4",
            )
        self._tokenizer = AutoTokenizer.from_pretrained(self.spec.model_id, token=token)
        self._model = AutoModelForCausalLM.from_pretrained(
            self.spec.model_id, token=token, **kwargs
        )
        if self.adapter_path:
            from peft import PeftModel
            self._model = PeftModel.from_pretrained(self._model, self.adapter_path)
        self._model.eval()

    # ------------------------------------------------------------------ #
    # Prompt rendering
    # ------------------------------------------------------------------ #
    def _render_base_transcript(self, messages: list[Message], prefill: str = "") -> str:
        """Render a plain-text transcript for a non-chat (pretrained) model."""
        lines: list[str] = []
        for m in messages:
            if m["role"] == "system":
                lines.append(m["content"])
            elif m["role"] == "user":
                lines.append(f"{_BASE_USER_TAG} {m['content']}")
            elif m["role"] == "assistant":
                lines.append(f"{_BASE_ASSISTANT_TAG} {m['content']}")
        # Open a fresh assistant turn (optionally prefilled).
        lines.append(f"{_BASE_ASSISTANT_TAG} {prefill}".rstrip() if not prefill
                     else f"{_BASE_ASSISTANT_TAG} {prefill}")
        return "\n".join(lines)

    def _build_inputs(self, messages: list[Message], prefill: str | None):
        """Return tokenized inputs for either chat or base models."""
        import torch  # noqa: F401  (ensures torch is importable before generate)

        if self.spec.is_base:
            text = self._render_base_transcript(messages, prefill or "")
            return self._tokenizer(text, return_tensors="pt").to(self._model.device)

        # Instruct model: use the chat template.
        if prefill is not None:
            msgs = list(messages) + [{"role": "assistant", "content": prefill}]
            ids = self._tokenizer.apply_chat_template(
                msgs,
                add_generation_prompt=False,
                continue_final_message=True,
                return_tensors="pt",
            )
        else:
            ids = self._tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_tensors="pt",
            )
        return {"input_ids": ids.to(self._model.device)}

    # ------------------------------------------------------------------ #
    # Generation
    # ------------------------------------------------------------------ #
    def _generate(self, messages, prefill, temperature, max_new_tokens) -> str:
        import torch

        self._ensure_loaded()
        inputs = self._build_inputs(messages, prefill)
        prompt_len = inputs["input_ids"].shape[-1]
        do_sample = temperature and temperature > 0
        with torch.no_grad():
            out = self._model.generate(
                **inputs,
                do_sample=do_sample,
                temperature=temperature if do_sample else None,
                top_p=1.0 if do_sample else None,
                max_new_tokens=max_new_tokens,
                pad_token_id=self._tokenizer.pad_token_id or self._tokenizer.eos_token_id,
            )
        gen = out[0][prompt_len:]
        return self._tokenizer.decode(gen, skip_special_tokens=True).strip()

    def chat(self, messages, *, temperature=config.TEMPERATURE,
             max_new_tokens=config.MAX_NEW_TOKENS) -> str:
        return self._generate(messages, None, temperature, max_new_tokens)

    def continue_response(self, messages, prefill, *, temperature=config.TEMPERATURE,
                          max_new_tokens=config.MAX_NEW_TOKENS) -> str:
        # The continuation excludes the prefill by construction (we decode only
        # the newly generated tokens after the prompt, which already contains it).
        return self._generate(messages, prefill, temperature, max_new_tokens)
