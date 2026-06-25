"""Unified chat client for target models across three backends.

Backends:
  - "google":     Gemini + Gemma via the google-genai SDK (Gemini API / AI Studio).
                  Requires GOOGLE_API_KEY (or GEMINI_API_KEY).
  - "openrouter": OpenAI-compatible OpenRouter endpoint (the paper's API path
                  for closed models). Requires OPENROUTER_API_KEY.
  - "hf":         local HuggingFace transformers (the paper's path for Gemma).
                  Requires transformers + torch + model weights.

All backends expose `.chat(messages, temperature, max_tokens) -> str`, where
`messages` is a list of {"role": "user"|"assistant", "content": str}. The core
elicitation eval uses no system prompt (Gemma 3 has no system role), matching
the paper. Thinking/reasoning is disabled best-effort (Section B.1 notes that
Gemini-2.5-Pro may still produce hidden reasoning regardless).
"""

from __future__ import annotations

import os
import time
from typing import Optional

from config import ModelSpec


class GenerationError(RuntimeError):
    pass


def _retry(fn, *, attempts: int = 5, base_delay: float = 2.0):
    """Run fn() with exponential backoff on transient errors."""
    last = None
    for i in range(attempts):
        try:
            return fn()
        except Exception as e:  # noqa: BLE001 - backends raise heterogeneous errors
            last = e
            if i == attempts - 1:
                break
            time.sleep(base_delay * (2 ** i))
    raise GenerationError(f"generation failed after {attempts} attempts: {last}") from last


# --------------------------------------------------------------------------
# Backend implementations
# --------------------------------------------------------------------------


class _GoogleBackend:
    """Gemini + Gemma via google-genai."""

    def __init__(self, spec: ModelSpec):
        from google import genai  # type: ignore

        api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise GenerationError("GOOGLE_API_KEY / GEMINI_API_KEY not set for google backend")
        self.spec = spec
        self.genai = genai
        self.client = genai.Client(api_key=api_key)
        self._is_gemma = spec.model_id.startswith("gemma")

    def _to_contents(self, messages: list[dict]):
        from google.genai import types  # type: ignore

        contents = []
        for m in messages:
            # google-genai uses "model" for assistant turns.
            role = "model" if m["role"] == "assistant" else "user"
            contents.append(types.Content(role=role, parts=[types.Part(text=m["content"])]))
        return contents

    def chat(self, messages: list[dict], temperature: float, max_tokens: int) -> str:
        from google.genai import types  # type: ignore

        cfg_kwargs = dict(temperature=temperature, max_output_tokens=max_tokens)
        # Disable thinking where supported. Gemma has no thinking; Gemini-2.5-Flash
        # supports thinking_budget=0; Pro cannot be fully disabled (per the paper).
        if self.spec.disable_thinking and not self._is_gemma:
            try:
                cfg_kwargs["thinking_config"] = types.ThinkingConfig(thinking_budget=0)
            except Exception:
                pass
        config = types.GenerateContentConfig(**cfg_kwargs)
        contents = self._to_contents(messages)

        def _call():
            resp = self.client.models.generate_content(
                model=self.spec.model_id, contents=contents, config=config
            )
            return resp.text or ""

        return _retry(_call)


class _OpenRouterBackend:
    """Any model via OpenRouter's OpenAI-compatible API."""

    def __init__(self, spec: ModelSpec):
        from openai import OpenAI  # type: ignore

        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise GenerationError("OPENROUTER_API_KEY not set for openrouter backend")
        self.spec = spec
        self.client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)

    def chat(self, messages: list[dict], temperature: float, max_tokens: int) -> str:
        extra_body = {}
        if self.spec.disable_thinking:
            # OpenRouter passes provider-specific reasoning controls through here.
            extra_body["reasoning"] = {"enabled": False}

        def _call():
            resp = self.client.chat.completions.create(
                model=self.spec.model_id,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                extra_body=extra_body or None,
            )
            return resp.choices[0].message.content or ""

        return _retry(_call)


class _HFBackend:
    """Local HuggingFace transformers inference (primarily for Gemma)."""

    _cache: dict = {}

    def __init__(self, spec: ModelSpec):
        self.spec = spec
        self.model, self.tokenizer = self._load(spec.model_id)

    @classmethod
    def _load(cls, model_id: str):
        if model_id in cls._cache:
            return cls._cache[model_id]
        import torch  # type: ignore
        from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore

        tok = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype="auto", device_map="auto"
        )
        cls._cache[model_id] = (model, tok)
        return model, tok

    def chat(self, messages: list[dict], temperature: float, max_tokens: int) -> str:
        import torch  # type: ignore

        inputs = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt"
        ).to(self.model.device)

        def _call():
            with torch.no_grad():
                out = self.model.generate(
                    inputs,
                    max_new_tokens=max_tokens,
                    do_sample=temperature > 0,
                    temperature=max(temperature, 1e-6),
                    top_p=1.0,
                )
            gen = out[0][inputs.shape[1]:]
            return self.tokenizer.decode(gen, skip_special_tokens=True)

        return _retry(_call, attempts=2, base_delay=1.0)


_BACKENDS = {"google": _GoogleBackend, "openrouter": _OpenRouterBackend, "hf": _HFBackend}


class ModelClient:
    """Thin façade selecting the right backend for a ModelSpec."""

    def __init__(self, spec: ModelSpec):
        if spec.backend not in _BACKENDS:
            raise ValueError(f"unknown backend: {spec.backend}")
        self.spec = spec
        self._backend = _BACKENDS[spec.backend](spec)

    def chat(self, messages: list[dict], temperature: float, max_tokens: int) -> str:
        return self._backend.chat(messages, temperature, max_tokens)
