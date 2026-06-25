"""Local HuggingFace inference for Gemma (instruct + base), with prefill support.

Loaded lazily and cached per (model_id, adapter) so multiple eval conditions
reuse the same weights.  A LoRA adapter path can be supplied to evaluate the
DPO/SFT finetunes (Section 4).
"""
from __future__ import annotations

import threading
from functools import lru_cache

from .base import ChatClient, GenConfig, Message

_LOAD_LOCK = threading.Lock()


@lru_cache(maxsize=4)
def _load_model_and_tokenizer(model_id: str, adapter_path: str | None, dtype: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch_dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16}.get(dtype, torch.bfloat16)
    with _LOAD_LOCK:
        tok = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=torch_dtype, device_map="auto"
        )
        if adapter_path:
            from peft import PeftModel

            model = PeftModel.from_pretrained(model, adapter_path)
        model.eval()
    return model, tok


class HFClient(ChatClient):
    supports_prefill = True

    def __init__(self, model_id: str, name: str | None = None, *,
                 adapter_path: str | None = None, is_base: bool = False,
                 dtype: str = "bfloat16"):
        super().__init__(model_id, name)
        self.adapter_path = adapter_path
        self.is_base = is_base
        self.dtype = dtype
        self._model = None
        self._tok = None

    def _ensure_loaded(self):
        if self._model is None:
            self._model, self._tok = _load_model_and_tokenizer(
                self.model_id, self.adapter_path, self.dtype
            )

    # -- tokenization helpers -------------------------------------------------
    def count_tokens(self, text: str) -> int:
        self._ensure_loaded()
        return len(self._tok.encode(text, add_special_tokens=False))

    def truncate_tokens(self, text: str, n_tokens: int) -> str:
        self._ensure_loaded()
        ids = self._tok.encode(text, add_special_tokens=False)[:n_tokens]
        return self._tok.decode(ids, skip_special_tokens=True)

    # -- prompt construction --------------------------------------------------
    def _build_inputs(self, messages: list[Message], prefill: str | None):
        """Return tokenised inputs.

        For instruct models we use the chat template.  For base models there is
        no chat template, so we render a lightweight transcript and rely on the
        prefill to anchor the assistant turn (Section 3 only ever calls base
        models with a prefill).
        """
        import torch

        if self.is_base:
            text = self._render_base_transcript(messages)
            if prefill:
                text += prefill
            enc = self._tok(text, return_tensors="pt", add_special_tokens=True)
        else:
            msgs = [{"role": m.role, "content": m.content} for m in messages]
            text = self._tok.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True
            )
            if prefill:
                text += prefill
            enc = self._tok(text, return_tensors="pt", add_special_tokens=False)
        return {k: v.to(self._model.device) for k, v in enc.items()}, text

    @staticmethod
    def _render_base_transcript(messages: list[Message]) -> str:
        parts = []
        for m in messages:
            tag = {"user": "User", "assistant": "Assistant", "system": "System"}[m.role]
            parts.append(f"{tag}: {m.content}")
        parts.append("Assistant:")
        return "\n\n".join(parts) + " "

    # -- generation -----------------------------------------------------------
    def generate(self, messages: list[Message], cfg: GenConfig,
                 prefill: str | None = None) -> str:
        import torch

        self._ensure_loaded()
        inputs, rendered = self._build_inputs(messages, prefill)
        gen_kwargs = dict(
            max_new_tokens=cfg.max_new_tokens,
            do_sample=cfg.temperature > 0,
            temperature=cfg.temperature,
            top_p=cfg.top_p,
            pad_token_id=self._tok.pad_token_id or self._tok.eos_token_id,
        )
        with torch.no_grad():
            out = self._model.generate(**inputs, **gen_kwargs)
        new_tokens = out[0][inputs["input_ids"].shape[1]:]
        continuation = self._tok.decode(new_tokens, skip_special_tokens=True)
        # By convention we return prefill + continuation so callers can strip the
        # prefill themselves when scoring "continuation only".
        return (prefill or "") + continuation
