"""Local Gemma backend (HuggingFace transformers).

Handles both instruct (`-it`) and base (`-pt`) checkpoints, and supports the
prefilled-continuation path used by the Section 3 base-vs-instruct experiment.

A single `GemmaClient` lazily loads the model/tokenizer once and reuses them.
LoRA adapters (Section 4) can be attached via `adapter_path`.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from .base import ChatMessage, ModelClient

if TYPE_CHECKING:
    from config import ModelSpec

log = logging.getLogger(__name__)


class GemmaClient(ModelClient):
    def __init__(
        self,
        spec: "ModelSpec",
        device: str | None = None,
        dtype: str = "bfloat16",
        adapter_path: str | None = None,
    ):
        self.spec = spec
        self.adapter_path = adapter_path
        self._device = device
        self._dtype = dtype
        self._model = None
        self._tok = None

    # ----- lazy loading ---------------------------------------------------- #
    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        dtype = getattr(torch, self._dtype)
        device_map = self._device or "auto"
        log.info("loading %s (adapter=%s)", self.spec.model_id, self.adapter_path)

        self._tok = AutoTokenizer.from_pretrained(self.spec.model_id)
        self._model = AutoModelForCausalLM.from_pretrained(
            self.spec.model_id, torch_dtype=dtype, device_map=device_map,
        )
        if self.adapter_path:
            from peft import PeftModel
            self._model = PeftModel.from_pretrained(self._model, self.adapter_path)
        self._model.eval()

    # ----- helpers --------------------------------------------------------- #
    def _render_chat(self, messages: list[ChatMessage], add_generation_prompt=True) -> str:
        """Render messages with the model's chat template.

        Gemma's chat template has no dedicated system role, so a leading system
        message is folded into the first user turn (documented in DESIGN.md).
        """
        msgs = list(messages)
        if msgs and msgs[0]["role"] == "system":
            sys = msgs[0]["content"]
            rest = msgs[1:]
            if rest and rest[0]["role"] == "user":
                rest = [{"role": "user", "content": f"{sys}\n\n{rest[0]['content']}"}] + rest[1:]
            else:
                rest = [{"role": "user", "content": sys}] + rest
            msgs = rest
        return self._tok.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=add_generation_prompt
        )

    def _generate(self, prompt_text: str, n: int, temperature: float, max_new_tokens: int,
                  stop: list[str] | None = None) -> list[str]:
        import torch

        inputs = self._tok(prompt_text, return_tensors="pt").to(self._model.device)
        gen_kwargs = dict(
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0,
            temperature=max(temperature, 1e-5),
            num_return_sequences=n,
            pad_token_id=self._tok.pad_token_id or self._tok.eos_token_id,
        )
        with torch.no_grad():
            out = self._model.generate(**inputs, **gen_kwargs)
        prompt_len = inputs["input_ids"].shape[1]
        texts = [self._tok.decode(o[prompt_len:], skip_special_tokens=True) for o in out]
        if stop:
            texts = [_truncate_at_stop(t, stop) for t in texts]
        return texts

    # ----- ModelClient API ------------------------------------------------- #
    def chat(self, messages, temperature=1.0, max_new_tokens=1024, stop=None) -> str:
        self._ensure_loaded()
        is_base = self.spec.is_base
        if is_base:
            # Base models aren't chat-tuned; concatenate turns plainly.
            prompt = _plain_concat(messages)
        else:
            prompt = self._render_chat(messages, add_generation_prompt=True)
        return self._generate(prompt, 1, temperature, max_new_tokens, stop)[0].strip()

    @property
    def supports_prefill(self) -> bool:
        return True

    def continue_from(self, messages, prefill, n=1, temperature=1.0, max_new_tokens=1024):
        self._ensure_loaded()
        if self.spec.is_base:
            base = _plain_concat(messages)
        else:
            base = self._render_chat(messages, add_generation_prompt=True)
        prompt = base + prefill
        return self._generate(prompt, n, temperature, max_new_tokens)


def _plain_concat(messages: list[ChatMessage]) -> str:
    """Plain-text rendering for base (non-chat) models."""
    parts = []
    for m in messages:
        if m["role"] == "system":
            parts.append(m["content"])
        elif m["role"] == "user":
            parts.append(f"User: {m['content']}")
        else:
            parts.append(f"Assistant: {m['content']}")
    parts.append("Assistant:")
    return "\n".join(parts)


def _truncate_at_stop(text: str, stop: list[str]) -> str:
    idxs = [text.find(s) for s in stop if text.find(s) != -1]
    return text[: min(idxs)] if idxs else text
