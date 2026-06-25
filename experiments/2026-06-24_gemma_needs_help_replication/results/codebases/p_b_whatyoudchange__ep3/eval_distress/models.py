"""Model clients.

Two backends:
  * HFModel        — local Gemma via HuggingFace transformers. Supports chat
                     generation, base-model prefill continuation, and exposing
                     hidden states / logits for the Section 3 + Appendix I work.
  * OpenRouterModel — Gemini (and other API models) via the OpenRouter
                     OpenAI-compatible endpoint. Generation only; thinking is
                     disabled per the paper.

Both implement a common `chat()` interface returning the assistant's text for a
list of {role, content} messages, plus generation kwargs (temperature, etc.).
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

from . import config_proxy as C


# ---------------------------------------------------------------------------
# Common message type
# ---------------------------------------------------------------------------
Message = dict  # {"role": "user"|"assistant"|"system", "content": str}


# ---------------------------------------------------------------------------
# Local HuggingFace Gemma
# ---------------------------------------------------------------------------
class HFModel:
    """Local Gemma (instruct or base) via transformers.

    Loaded lazily so importing this module is cheap on machines without a GPU.
    For the 27B model on a single GPU, pass load_in_4bit=True.
    """

    def __init__(self, model_id: str, *, adapter_path: Optional[str] = None,
                 load_in_4bit: bool = False, device_map: str = "auto",
                 dtype: str = "bfloat16"):
        self.model_id = model_id
        self.adapter_path = adapter_path
        self.load_in_4bit = load_in_4bit
        self.device_map = device_map
        self.dtype = dtype
        self._model = None
        self._tok = None

    # -- lazy load ----------------------------------------------------------
    def _ensure_loaded(self):
        if self._model is not None:
            return
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tok = AutoTokenizer.from_pretrained(self.model_id)
        kwargs = {"device_map": self.device_map,
                  "torch_dtype": getattr(torch, self.dtype)}
        if self.load_in_4bit:
            from transformers import BitsAndBytesConfig
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4")
        model = AutoModelForCausalLM.from_pretrained(self.model_id, **kwargs)
        if self.adapter_path:
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, self.adapter_path)
        model.eval()
        self._model, self._tok = model, tok

    # -- helpers ------------------------------------------------------------
    def _render_chat(self, messages: list[Message], add_generation_prompt=True) -> str:
        """Render with Gemma's chat template. Gemma has no system role, so a
        leading system message is folded into the first user turn (the standard
        transformers behaviour for Gemma)."""
        self._ensure_loaded()
        msgs = self._fold_system(messages)
        return self._tok.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=add_generation_prompt)

    @staticmethod
    def _fold_system(messages: list[Message]) -> list[Message]:
        if messages and messages[0]["role"] == "system":
            sys_txt = messages[0]["content"]
            rest = messages[1:]
            if rest and rest[0]["role"] == "user":
                folded = dict(rest[0])
                folded["content"] = f"{sys_txt}\n\n{rest[0]['content']}"
                return [folded] + rest[1:]
            return [{"role": "user", "content": sys_txt}] + rest
        return messages

    # -- generation ---------------------------------------------------------
    def chat(self, messages: list[Message], *, temperature: float = C.TEMPERATURE,
             max_new_tokens: int = C.MAX_NEW_TOKENS, prefill: str = "") -> str:
        """Generate the assistant response to `messages`.

        If `prefill` is given, the assistant turn is started with that text and
        the model continues it (used for the Section 3 prefill experiment and
        for base models, which are not chat-tuned)."""
        import torch
        self._ensure_loaded()
        prompt = self._render_chat(messages, add_generation_prompt=True)
        if prefill:
            prompt = prompt + prefill
        inputs = self._tok(prompt, return_tensors="pt").to(self._model.device)
        with torch.no_grad():
            out = self._model.generate(
                **inputs,
                do_sample=temperature > 0,
                temperature=temperature if temperature > 0 else None,
                top_p=1.0,
                max_new_tokens=max_new_tokens,
                pad_token_id=self._tok.eos_token_id,
            )
        gen = out[0][inputs["input_ids"].shape[1]:]
        return self._tok.decode(gen, skip_special_tokens=True)

    def complete_raw(self, text: str, *, temperature: float = C.TEMPERATURE,
                     max_new_tokens: int = C.MAX_NEW_TOKENS) -> str:
        """Raw (non-chat) continuation of `text`. Used for base-model prefill
        continuation in Section 3, where we present the conversation as plain
        text and let the base model continue."""
        import torch
        self._ensure_loaded()
        inputs = self._tok(text, return_tensors="pt").to(self._model.device)
        with torch.no_grad():
            out = self._model.generate(
                **inputs,
                do_sample=temperature > 0,
                temperature=temperature if temperature > 0 else None,
                top_p=1.0,
                max_new_tokens=max_new_tokens,
                pad_token_id=self._tok.eos_token_id,
            )
        gen = out[0][inputs["input_ids"].shape[1]:]
        return self._tok.decode(gen, skip_special_tokens=True)

    # exposed for Appendix-I logit-lens work
    @property
    def model(self):
        self._ensure_loaded()
        return self._model

    @property
    def tokenizer(self):
        self._ensure_loaded()
        return self._tok


# ---------------------------------------------------------------------------
# OpenRouter (Gemini)
# ---------------------------------------------------------------------------
class OpenRouterModel:
    """Gemini via OpenRouter's OpenAI-compatible API. Thinking disabled per
    the paper ('we set thinking to be false via the API')."""

    def __init__(self, model_id: str):
        self.model_id = model_id
        self._client = None

    def _ensure_client(self):
        if self._client is not None:
            return
        from openai import OpenAI
        key = os.environ.get(C.OPENROUTER_API_KEY_ENV)
        if not key:
            raise RuntimeError(
                f"{C.OPENROUTER_API_KEY_ENV} not set; required for OpenRouter models.")
        self._client = OpenAI(base_url=C.OPENROUTER_BASE_URL, api_key=key)

    def chat(self, messages: list[Message], *, temperature: float = C.TEMPERATURE,
             max_new_tokens: int = C.MAX_NEW_TOKENS, prefill: str = "") -> str:
        if prefill:
            raise NotImplementedError(
                "OpenRouter/Gemini does not support assistant prefill; Gemini "
                "is excluded from the Section 3 prefill experiment.")
        self._ensure_client()
        # Disable thinking where the provider exposes the toggle.
        extra_body = {
            "reasoning": {"enabled": False},
            "provider": {"require_parameters": False},
        }
        resp = self._client.chat.completions.create(
            model=self.model_id,
            messages=messages,
            temperature=temperature,
            max_tokens=max_new_tokens,
            extra_body=extra_body,
        )
        return resp.choices[0].message.content or ""


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------
def load_target(model_key: str, *, adapter_path: Optional[str] = None,
                load_in_4bit: bool = False):
    spec = C.TARGET_MODELS[model_key]
    if spec.backend == "hf":
        return HFModel(spec.model_id, adapter_path=adapter_path,
                       load_in_4bit=load_in_4bit)
    if spec.backend == "openrouter":
        if adapter_path:
            raise ValueError("Adapters are only supported for local HF models.")
        return OpenRouterModel(spec.model_id)
    raise ValueError(f"Unknown backend {spec.backend!r}")
