"""Model-client abstraction.

A `ChatModel` exposes `.chat(messages, temperature, max_tokens)` returning the
assistant text. Three backends cover the paper's access patterns within the
Gemma+Gemini scope:

  OpenRouterModel  -- Gemini-2.5 Flash/Pro (and, optionally, Gemma) via the
                      OpenAI-compatible OpenRouter API. thinking disabled.
  HFModel          -- local HuggingFace transformers for Gemma weights. Also
                      exposes prefilled generation and raw logits, which the
                      OpenRouter path cannot provide (needed by Sections 3 & I).
  AnthropicModel   -- Claude judge / auditor (and the GPT validation judge runs
                      through OpenRouterModel).

The factory `get_model(spec)` dispatches on `spec.backend`. HF models are heavy,
so they are lazily constructed and cached per process.
"""
from __future__ import annotations

import functools
import os
import time
from typing import Any

from . import config
from .config import ModelSpec

Message = dict[str, str]   # {"role": "user"|"assistant"|"system", "content": ...}


# --------------------------------------------------------------------------- #
# Base interface
# --------------------------------------------------------------------------- #
class ChatModel:
    spec: ModelSpec

    def chat(self, messages: list[Message], *, temperature: float = config.TEMPERATURE,
             max_tokens: int = config.MAX_NEW_TOKENS, system: str | None = None) -> str:
        raise NotImplementedError

    # Optional capabilities; only HFModel implements them. Callers check via
    # `supports_prefill` / `supports_logits`.
    supports_prefill: bool = False
    supports_logits: bool = False

    def chat_prefilled(self, messages: list[Message], prefill: str, *,
                       temperature: float = config.TEMPERATURE,
                       max_tokens: int = config.MAX_NEW_TOKENS,
                       system: str | None = None) -> str:
        raise NotImplementedError("backend does not support prefilled generation")

    def complete(self, text: str, *, temperature: float = config.TEMPERATURE,
                 max_tokens: int = config.MAX_NEW_TOKENS) -> str:
        """Raw text completion (no chat template) -- for base models."""
        raise NotImplementedError("backend does not support raw completion")


def _retry(fn, *, tries: int = 5, base: float = 2.0):
    last = None
    for i in range(tries):
        try:
            return fn()
        except Exception as e:    # noqa: BLE001 -- provider exceptions vary
            last = e
            time.sleep(base * (2 ** i))
    raise last


# --------------------------------------------------------------------------- #
# OpenRouter (Gemini, optional Gemma, GPT validation judge)
# --------------------------------------------------------------------------- #
class OpenRouterModel(ChatModel):
    def __init__(self, spec: ModelSpec):
        from openai import OpenAI
        self.spec = spec
        self._client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=config.require_key("OPENROUTER_API_KEY"),
        )

    def _extra_body(self) -> dict[str, Any]:
        """Provider-specific fields. The OpenAI SDK forwards everything under
        `extra_body` verbatim into the request JSON, which is how OpenRouter
        receives non-standard params."""
        if config.DISABLE_THINKING and self.spec.family == "gemini":
            # Disable Gemini reasoning two ways (OpenRouter's unified toggle and
            # Google's native thinking_budget). Per the paper, Gemini-2.5-Pro may
            # still emit hidden reasoning the API cannot fully suppress.
            return {
                "reasoning": {"enabled": False},
                "google": {"thinking_config": {"thinking_budget": 0}},
            }
        return {}

    def chat(self, messages, *, temperature=config.TEMPERATURE,
             max_tokens=config.MAX_NEW_TOKENS, system=None) -> str:
        msgs = ([{"role": "system", "content": system}] if system else []) + messages

        def call():
            return self._client.chat.completions.create(
                model=self.spec.model_id,
                messages=msgs,
                temperature=temperature,
                max_tokens=max_tokens,
                extra_body=self._extra_body(),
            )

        resp = _retry(call)
        return (resp.choices[0].message.content or "").strip()


# --------------------------------------------------------------------------- #
# Anthropic (judge / auditor)
# --------------------------------------------------------------------------- #
class AnthropicModel(ChatModel):
    def __init__(self, spec: ModelSpec):
        import anthropic
        self.spec = spec
        self._client = anthropic.Anthropic(api_key=config.require_key("ANTHROPIC_API_KEY"))

    def chat(self, messages, *, temperature=config.TEMPERATURE,
             max_tokens=config.MAX_NEW_TOKENS, system=None) -> str:
        def call():
            kwargs: dict[str, Any] = dict(
                model=self.spec.model_id,
                max_tokens=max_tokens,
                temperature=temperature,
                messages=messages,
            )
            if system:
                kwargs["system"] = system
            return self._client.messages.create(**kwargs)

        resp = _retry(call)
        return "".join(b.text for b in resp.content if b.type == "text").strip()


# --------------------------------------------------------------------------- #
# Local HuggingFace transformers (Gemma): chat, prefilled chat, raw completion.
# --------------------------------------------------------------------------- #
class HFModel(ChatModel):
    supports_prefill = True
    supports_logits = True

    def __init__(self, spec: ModelSpec, *, load_in_4bit: bool | None = None,
                 adapter_path: str | None = None, dtype: str = "bfloat16"):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.spec = spec
        self._torch = torch
        token = os.environ.get("HF_TOKEN")
        if load_in_4bit is None:
            load_in_4bit = "27b" in spec.model_id.lower()    # 27B needs it on 1 GPU

        quant = None
        if load_in_4bit:
            from transformers import BitsAndBytesConfig
            quant = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=getattr(torch, dtype),
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )

        self.tokenizer = AutoTokenizer.from_pretrained(spec.model_id, token=token)
        self.model = AutoModelForCausalLM.from_pretrained(
            spec.model_id,
            token=token,
            torch_dtype=getattr(torch, dtype),
            device_map="auto",
            quantization_config=quant,
            attn_implementation="eager",   # Gemma-3 recommends eager attention
        )
        if adapter_path:
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, adapter_path)
        self.model.eval()

    # -- helpers ----------------------------------------------------------- #
    def _render(self, messages, system, add_generation_prompt=True,
                continue_final=False) -> str:
        msgs = ([{"role": "system", "content": system}] if system else []) + messages
        return self.tokenizer.apply_chat_template(
            msgs, tokenize=False,
            add_generation_prompt=add_generation_prompt,
            continue_final_message=continue_final,
        )

    def _generate(self, prompt_text: str, temperature: float, max_tokens: int) -> str:
        torch = self._torch
        inputs = self.tokenizer(prompt_text, return_tensors="pt",
                                add_special_tokens=False).to(self.model.device)
        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                do_sample=temperature > 0,
                temperature=temperature if temperature > 0 else None,
                top_p=1.0 if temperature > 0 else None,
                max_new_tokens=max_tokens,
                pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
            )
        gen = out[0][inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(gen, skip_special_tokens=True).strip()

    # -- ChatModel API ----------------------------------------------------- #
    def chat(self, messages, *, temperature=config.TEMPERATURE,
             max_tokens=config.MAX_NEW_TOKENS, system=None) -> str:
        return self._generate(self._render(messages, system), temperature, max_tokens)

    def chat_prefilled(self, messages, prefill, *, temperature=config.TEMPERATURE,
                       max_tokens=config.MAX_NEW_TOKENS, system=None) -> str:
        """Force the assistant turn to start with `prefill`, return the *whole*
        assistant turn (prefill + continuation). Callers strip the prefill."""
        msgs = list(messages) + [{"role": "assistant", "content": prefill}]
        text = self._render(msgs, system, add_generation_prompt=False,
                            continue_final=True)
        cont = self._generate(text, temperature, max_tokens)
        return prefill + cont

    def complete(self, text, *, temperature=config.TEMPERATURE,
                 max_tokens=config.MAX_NEW_TOKENS) -> str:
        """Raw completion for base/pretrained models (no chat template)."""
        return self._generate(text, temperature, max_tokens)


# --------------------------------------------------------------------------- #
# Factory
# --------------------------------------------------------------------------- #
@functools.lru_cache(maxsize=None)
def get_model(spec_key_or_spec) -> ChatModel:
    # Registered LoRA adapters take precedence over plain specs.
    if isinstance(spec_key_or_spec, str) and spec_key_or_spec in _ADAPTERS:
        adapter_path, base_key = _ADAPTERS[spec_key_or_spec]
        return load_finetuned(adapter_path, base_key)
    spec = (spec_key_or_spec if isinstance(spec_key_or_spec, ModelSpec)
            else _resolve_spec(spec_key_or_spec))
    if spec.backend == "openrouter":
        return OpenRouterModel(spec)
    if spec.backend == "anthropic":
        return AnthropicModel(spec)
    if spec.backend == "hf":
        return HFModel(spec)
    raise ValueError(f"unknown backend: {spec.backend}")


def _resolve_spec(key: str) -> ModelSpec:
    for registry in (config.MODELS,):
        if key in registry:
            return registry[key]
    for special in (config.JUDGE_MODEL, config.PETRI_AUDITOR_MODEL,
                    config.PETRI_JUDGE_MODEL, config.VALIDATION_JUDGE_MODEL):
        if special.key == key or special.model_id == key:
            return special
    raise KeyError(f"no model spec registered for {key!r}")


def load_finetuned(adapter_path: str, base_key: str = "gemma-3-27b-it") -> HFModel:
    """Load a base Gemma with a trained LoRA adapter (DPO/SFT/ablation)."""
    base_spec = config.MODELS[base_key]
    return HFModel(base_spec, adapter_path=adapter_path)


# --------------------------------------------------------------------------- #
# Adapter registry: lets finetuned/ablated Gemma variants be referenced by a
# short key throughout the pipeline (runner, analysis, etc.), exactly like the
# built-in models, without baking adapter paths into config.MODELS.
# --------------------------------------------------------------------------- #
_ADAPTERS: dict[str, tuple[str, str]] = {}   # key -> (adapter_path, base_key)


def register_finetuned(key: str, adapter_path: str,
                       base_key: str = "gemma-3-27b-it") -> None:
    """Register a LoRA adapter under `key`. Subsequent get_model(key) returns
    the base model with the adapter applied."""
    _ADAPTERS[key] = (str(adapter_path), base_key)
    get_model.cache_clear()    # ensure the new key resolves freshly
