"""Target-model client abstraction (the models under test).

Three backends are supported:
  - OpenRouterModel: HTTP to OpenRouter's chat-completions API. Works for
    every model with one key (OPENROUTER_API_KEY). Default.
  - HuggingFaceModel: local transformers inference. Paper-faithful path for
    Gemma (needs a GPU + `transformers`/`torch`).
  - GoogleGenAIModel: native Gemini via google-genai (GOOGLE_API_KEY).

All backends expose `.chat(messages, temperature, max_tokens) -> str` where
`messages` is a list of {"role": "user"|"assistant", "content": str}. Model
"thinking"/reasoning is disabled where the backend allows it (Section 2.1).
"""

from __future__ import annotations

import os
import time
from typing import Protocol

import config


class TargetModel(Protocol):
    """A chat model under test."""

    name: str

    def chat(self, messages: list[dict], temperature: float,
             max_tokens: int) -> str:
        ...


# ---------------------------------------------------------------------------
# OpenRouter backend.
# ---------------------------------------------------------------------------
class OpenRouterModel:
    """Chat via OpenRouter's OpenAI-compatible API."""

    ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"

    def __init__(self, name: str, model_id: str,
                 disable_thinking: bool = True,
                 api_key: str | None = None,
                 max_retries: int = 5):
        self.name = name
        self.model_id = model_id
        self.disable_thinking = disable_thinking
        self.max_retries = max_retries
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
        if not self.api_key:
            raise RuntimeError(
                "OPENROUTER_API_KEY is not set (required for OpenRouter backend)."
            )

    def _payload(self, messages, temperature, max_tokens, with_reasoning):
        payload = {
            "model": self.model_id,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if with_reasoning and self.disable_thinking:
            # OpenRouter unifies reasoning control under `reasoning`. Disabling
            # it suppresses thinking for models that support it (Gemini). Gemma
            # has no thinking; the flag is harmless there. NB: Gemini 2.5 Pro
            # may still emit hidden reasoning (paper Appendix B.1).
            payload["reasoning"] = {"enabled": False}
        return payload

    def chat(self, messages, temperature, max_tokens) -> str:
        import requests  # local import keeps optional deps lazy

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        last_err: Exception | None = None
        with_reasoning = True
        for attempt in range(self.max_retries):
            try:
                payload = self._payload(messages, temperature, max_tokens,
                                        with_reasoning)
                resp = requests.post(self.ENDPOINT, headers=headers,
                                     json=payload, timeout=180)
                if resp.status_code == 400 and with_reasoning:
                    # Some routes reject the `reasoning` field; retry without it.
                    with_reasoning = False
                    continue
                if resp.status_code in (429, 500, 502, 503, 529):
                    raise _Retryable(f"HTTP {resp.status_code}: {resp.text[:200]}")
                resp.raise_for_status()
                data = resp.json()
                return data["choices"][0]["message"]["content"] or ""
            except _Retryable as exc:
                last_err = exc
            except Exception as exc:  # noqa: BLE001
                last_err = exc
            _backoff(attempt)
        raise RuntimeError(
            f"OpenRouter request failed after {self.max_retries} attempts: {last_err}"
        )


# ---------------------------------------------------------------------------
# Google GenAI backend (native Gemini).
# ---------------------------------------------------------------------------
class GoogleGenAIModel:
    """Chat via the native Gemini API (google-genai)."""

    def __init__(self, name: str, model_id: str,
                 disable_thinking: bool = True,
                 api_key: str | None = None,
                 max_retries: int = 5):
        self.name = name
        self.model_id = model_id
        self.disable_thinking = disable_thinking
        self.max_retries = max_retries
        api_key = api_key or os.environ.get("GOOGLE_API_KEY") \
            or os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "GOOGLE_API_KEY (or GEMINI_API_KEY) is not set for the google backend."
            )
        from google import genai  # local import: optional dependency

        self._genai = genai
        self._client = genai.Client(api_key=api_key)

    @staticmethod
    def _to_contents(messages):
        # Gemini uses roles "user" and "model".
        contents = []
        for m in messages:
            role = "model" if m["role"] == "assistant" else "user"
            contents.append({"role": role, "parts": [{"text": m["content"]}]})
        return contents

    def chat(self, messages, temperature, max_tokens) -> str:
        from google.genai import types

        thinking = None
        if self.disable_thinking:
            # thinking_budget=0 disables thinking where supported (Flash). Pro
            # may ignore this and still reason internally (paper note).
            try:
                thinking = types.ThinkingConfig(thinking_budget=0)
            except Exception:  # noqa: BLE001
                thinking = None
        cfg = types.GenerateContentConfig(
            temperature=temperature,
            max_output_tokens=max_tokens,
            thinking_config=thinking,
        )
        last_err: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                resp = self._client.models.generate_content(
                    model=self.model_id,
                    contents=self._to_contents(messages),
                    config=cfg,
                )
                return resp.text or ""
            except Exception as exc:  # noqa: BLE001
                last_err = exc
                _backoff(attempt)
        raise RuntimeError(
            f"Gemini request failed after {self.max_retries} attempts: {last_err}"
        )


# ---------------------------------------------------------------------------
# HuggingFace local backend (paper-faithful path for Gemma).
# ---------------------------------------------------------------------------
class HuggingFaceModel:
    """Local transformers inference. Loads the model once and reuses it.

    Generation is GPU-bound; run with --max-workers 1 so a single model
    instance handles rollouts sequentially.
    """

    def __init__(self, name: str, model_id: str,
                 disable_thinking: bool = True,
                 device: str | None = None, dtype: str = "bfloat16"):
        self.name = name
        self.model_id = model_id
        self.disable_thinking = disable_thinking  # Gemma has no thinking mode
        import torch  # local import: optional dependency
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=getattr(torch, dtype),
            device_map=device or "auto",
        )
        self.model.eval()

    def chat(self, messages, temperature, max_tokens) -> str:
        torch = self._torch
        # Gemma chat template does not support a system role; we never use one.
        inputs = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to(self.model.device)
        with torch.no_grad():
            out = self.model.generate(
                inputs,
                max_new_tokens=max_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=1.0,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        new_tokens = out[0][inputs.shape[-1]:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True)


# ---------------------------------------------------------------------------
# Factory + helpers.
# ---------------------------------------------------------------------------
class _Retryable(Exception):
    pass


def _backoff(attempt: int) -> None:
    time.sleep(min(2 ** attempt, 30))


def build_model(model_key: str, provider: str | None = None) -> TargetModel:
    """Construct a TargetModel for `model_key` using the given provider.

    `provider` defaults to config.DEFAULT_PROVIDER[model_key].
    """
    if model_key not in config.MODELS:
        raise ValueError(f"Unknown model: {model_key}. Known: {config.ALL_MODELS}")
    provider = provider or config.DEFAULT_PROVIDER[model_key]
    backends = config.MODELS[model_key]
    if provider not in backends:
        raise ValueError(
            f"Provider '{provider}' not available for {model_key}. "
            f"Available: {sorted(backends)}"
        )
    model_id = backends[provider]
    if provider == "openrouter":
        return OpenRouterModel(model_key, model_id,
                               disable_thinking=config.DISABLE_THINKING)
    if provider == "google":
        return GoogleGenAIModel(model_key, model_id,
                                disable_thinking=config.DISABLE_THINKING)
    if provider == "huggingface":
        return HuggingFaceModel(model_key, model_id,
                                disable_thinking=config.DISABLE_THINKING)
    raise ValueError(f"Unknown provider: {provider}")
