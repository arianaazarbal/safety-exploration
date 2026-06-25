"""Model generation backends.

Two backends cover the in-scope models:

* ``HFLocalProvider``  - Gemma (open weights) via 🤗 transformers, optionally
  with a LoRA adapter on top (our DPO/SFT models). Runs the chat template for
  instruct models; supports raw-continuation generation for base models and the
  prefill experiments.
* ``OpenRouterProvider`` - Gemini via OpenRouter's OpenAI-compatible API.

Both expose a uniform ``chat(messages, ...) -> str`` and the HF backend adds
``continue_text(prefix, ...)`` for prefill/continuation experiments.

All generation uses temperature=1 (paper default). Thinking is disabled for the
API models per Appendix B.1.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import config
from config import ModelSpec


# --------------------------------------------------------------------------- #
# Message type: list of {"role": "user"|"assistant"|"system", "content": str}
# --------------------------------------------------------------------------- #
Message = dict[str, str]


class BaseProvider:
    def chat(self, messages: list[Message], *, max_new_tokens: int | None = None,
             temperature: float = config.TEMPERATURE) -> str:
        raise NotImplementedError


# --------------------------------------------------------------------------- #
# HuggingFace local backend (Gemma)
# --------------------------------------------------------------------------- #
class HFLocalProvider(BaseProvider):
    def __init__(self, spec: ModelSpec, *, load_in_4bit: bool = False,
                 device_map: str = "auto", dtype: str = "bfloat16"):
        self.spec = spec
        self._load_in_4bit = load_in_4bit
        self._device_map = device_map
        self._dtype = dtype
        self._model = None
        self._tokenizer = None

    # Lazy load so importing this module never pulls weights.
    def _ensure_loaded(self):
        if self._model is not None:
            return
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        kwargs: dict[str, Any] = {
            "device_map": self._device_map,
            "torch_dtype": getattr(torch, self._dtype),
        }
        if self._load_in_4bit:
            from transformers import BitsAndBytesConfig

            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4",
            )
        self._tokenizer = AutoTokenizer.from_pretrained(self.spec.model_id)
        model = AutoModelForCausalLM.from_pretrained(self.spec.model_id, **kwargs)

        if self.spec.adapter_path and os.path.isdir(self.spec.adapter_path):
            from peft import PeftModel

            model = PeftModel.from_pretrained(model, self.spec.adapter_path)
            model = model.merge_and_unload()  # fold LoRA in for faster inference
        model.eval()
        self._model = model

    def _generate(self, input_ids, attention_mask, max_new_tokens, temperature):
        import torch

        do_sample = temperature and temperature > 0
        with torch.no_grad():
            out = self._model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens or config.MAX_NEW_TOKENS,
                do_sample=do_sample,
                temperature=temperature if do_sample else None,
                top_p=config.TOP_P if do_sample else None,
                pad_token_id=self._tokenizer.pad_token_id
                or self._tokenizer.eos_token_id,
            )
        gen = out[0][input_ids.shape[1]:]
        return self._tokenizer.decode(gen, skip_special_tokens=True)

    def chat(self, messages, *, max_new_tokens=None, temperature=config.TEMPERATURE):
        self._ensure_loaded()
        # Instruct models: apply the chat template. Base models: fall back to a
        # plain User:/Assistant: concatenation (they have no chat template).
        if self.spec.is_base:
            enc = self._tokenizer(_plainify(messages), return_tensors="pt").to(
                self._model.device)
            input_ids = enc["input_ids"]
        else:
            input_ids = self._tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, return_tensors="pt"
            ).to(self._model.device)
        pad_id = self._tokenizer.pad_token_id or 0
        attn = (input_ids != pad_id).long()
        return self._generate(input_ids, attn, max_new_tokens, temperature)

    def continue_text(self, messages, prefill, *, max_new_tokens=None,
                      temperature=config.TEMPERATURE):
        """Generate a continuation of `prefill` as the assistant's response to the
        conversation in `messages`.

        For instruct models this builds the chat-template prefix and appends the
        prefill (no generation prompt suffix that would close the turn); for base
        models it concatenates everything as raw text. Returns ONLY the newly
        generated continuation (excluding the prefill), matching the Section 3
        protocol where "the generated continuation (excluding prefill) is scored".
        """
        self._ensure_loaded()
        if self.spec.is_base:
            prefix = _plainify(messages) + "\n" + prefill
        else:
            prefix = self._tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False
            ) + prefill
        enc = self._tokenizer(prefix, return_tensors="pt").to(self._model.device)
        attn = enc.get("attention_mask")
        return self._generate(enc["input_ids"], attn, max_new_tokens, temperature)


def _plainify(messages: list[Message]) -> str:
    """Render a conversation as plain User:/Assistant: text for base models."""
    lines = []
    for m in messages:
        role = m["role"].capitalize()
        lines.append(f"{role}: {m['content']}")
    lines.append("Assistant:")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# OpenRouter backend (Gemini) — OpenAI-compatible
# --------------------------------------------------------------------------- #
class OpenRouterProvider(BaseProvider):
    def __init__(self, spec: ModelSpec):
        self.spec = spec
        self._client = None

    def _ensure_client(self):
        if self._client is not None:
            return
        from openai import OpenAI

        self._client = OpenAI(
            base_url=config.OPENROUTER_BASE_URL,
            api_key=config.OPENROUTER_API_KEY,
        )

    def chat(self, messages, *, max_new_tokens=None, temperature=config.TEMPERATURE):
        self._ensure_client()
        # Disable thinking/reasoning per Appendix B.1 ("we set thinking to be
        # false via the API"). OpenRouter passes `reasoning` through to Gemini.
        resp = self._client.chat.completions.create(
            model=self.spec.model_id,
            messages=messages,
            temperature=temperature,
            top_p=config.TOP_P,
            max_tokens=max_new_tokens or config.MAX_NEW_TOKENS,
            extra_body={"reasoning": {"enabled": False}},
        )
        return resp.choices[0].message.content or ""


# --------------------------------------------------------------------------- #
# Factory + cache
# --------------------------------------------------------------------------- #
_PROVIDER_CACHE: dict[str, BaseProvider] = {}


def get_provider(model_key: str, **kwargs) -> BaseProvider:
    """Return a (cached) provider for a model key from config.MODELS."""
    if model_key in _PROVIDER_CACHE:
        return _PROVIDER_CACHE[model_key]
    spec = config.MODELS[model_key]
    if spec.backend == "hf_local":
        provider: BaseProvider = HFLocalProvider(spec, **kwargs)
    elif spec.backend == "openrouter":
        provider = OpenRouterProvider(spec)
    else:
        raise ValueError(f"Unknown backend {spec.backend!r} for {model_key}")
    _PROVIDER_CACHE[model_key] = provider
    return provider
