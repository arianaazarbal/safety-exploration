"""Local HuggingFace transformers backend.

Used for everything that needs weights/logits rather than a chat API:
  - Gemma instruct + base inference (Section 2/3)
  - prefilled continuations (Section 3, Section 4.2 recovery)
  - loading LoRA adapters produced by the training scripts (Section 4)
  - exposing `.model` / `.tokenizer` for the logit-based probing (Appendix I)

Models are loaded lazily on first use so that importing this module is cheap on
machines without a GPU.
"""
from __future__ import annotations

from typing import Sequence

from ..config import ModelSpec
from .base import Message


def _build_base_transcript(messages: Sequence[Message]) -> str:
    """Render a chat transcript as plain text for a BASE (non-chat) model.

    Base models aren't trained on a chat template, so the paper relies on
    prefilling to get consistent continuations (Section 3.1). We use a simple,
    neutral "User:/Assistant:" rendering; the assistant prefix is appended by
    the caller via `prefill`.
    """
    parts = []
    for m in messages:
        role = m["role"]
        if role == "system":
            parts.append(m["content"])
        elif role == "user":
            parts.append(f"User: {m['content']}")
        elif role == "assistant":
            parts.append(f"Assistant: {m['content']}")
    parts.append("Assistant:")
    return "\n\n".join(parts) + " "


class LocalHFClient:
    def __init__(self, spec: ModelSpec, device: str | None = None, dtype: str = "bfloat16"):
        self.spec = spec
        self.name = spec.name
        self.device = device
        self.dtype = dtype
        self._model = None
        self._tokenizer = None

    # ----- lazy loading -----------------------------------------------------
    def _ensure_loaded(self):
        if self._model is not None:
            return
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        hf_id = self.spec.hf_id
        if not hf_id:
            raise ValueError(f"Model '{self.spec.name}' has no hf_id")
        torch_dtype = getattr(torch, self.dtype)
        self._tokenizer = AutoTokenizer.from_pretrained(hf_id)
        model = AutoModelForCausalLM.from_pretrained(
            hf_id,
            torch_dtype=torch_dtype,
            device_map=self.device or "auto",
        )
        if self.spec.adapter_path:
            from peft import PeftModel

            model = PeftModel.from_pretrained(model, self.spec.adapter_path)
        model.eval()
        self._model = model

    @property
    def model(self):
        self._ensure_loaded()
        return self._model

    @property
    def tokenizer(self):
        self._ensure_loaded()
        return self._tokenizer

    # ----- prompt construction ----------------------------------------------
    def _render_prompt(self, messages: Sequence[Message], add_generation_prompt: bool) -> str:
        if self.spec.chat_template:
            return self.tokenizer.apply_chat_template(
                list(messages),
                tokenize=False,
                add_generation_prompt=add_generation_prompt,
            )
        return _build_base_transcript(messages)

    # ----- generation -------------------------------------------------------
    def _generate(self, prompt: str, *, temperature: float, max_tokens: int) -> str:
        import torch

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        prompt_len = inputs["input_ids"].shape[1]
        do_sample = temperature is not None and temperature > 0
        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                do_sample=do_sample,
                temperature=temperature if do_sample else None,
                max_new_tokens=max_tokens,
                pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
            )
        new_tokens = out[0][prompt_len:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True)

    def chat(
        self,
        messages: Sequence[Message],
        *,
        temperature: float = 1.0,
        max_tokens: int = 2048,
        system: str | None = None,
    ) -> str:
        msgs = list(messages)
        if system:
            msgs = [{"role": "system", "content": system}, *msgs]
        prompt = self._render_prompt(msgs, add_generation_prompt=True)
        return self._generate(prompt, temperature=temperature, max_tokens=max_tokens)

    def continue_from(
        self,
        messages: Sequence[Message],
        prefill: str,
        *,
        temperature: float = 1.0,
        max_tokens: int = 1024,
    ) -> str:
        """Continue an assistant turn that begins with `prefill`.

        Returns only the newly generated text (the paper scores the continuation
        excluding the prefill).
        """
        prompt = self._render_prompt(messages, add_generation_prompt=True) + prefill
        return self._generate(prompt, temperature=temperature, max_tokens=max_tokens)
