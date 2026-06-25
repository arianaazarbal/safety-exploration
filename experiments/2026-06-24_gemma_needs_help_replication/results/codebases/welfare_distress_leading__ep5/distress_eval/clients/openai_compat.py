"""OpenAI-compatible chat backend.

Covers three of our configured backends, all of which speak the OpenAI
chat-completions wire format:

  * ``openrouter`` -> https://openrouter.ai/api/v1 (Gemma + Gemini targets,
    and optionally the GPT validation judge)
  * ``openai``     -> https://api.openai.com/v1 (GPT validation judge)
  * any custom base_url (self-hosted vLLM, LM Studio, etc.)

API key + base URL resolution is by backend name so a single config can mix
providers. Retries use exponential backoff on transient errors.
"""

from __future__ import annotations

import os
from typing import List

from tenacity import retry, stop_after_attempt, wait_exponential

from .base import Message

_BACKEND_DEFAULTS = {
    "openrouter": {
        "base_url": "https://openrouter.ai/api/v1",
        "key_env": "OPENROUTER_API_KEY",
    },
    "openai": {
        "base_url": "https://api.openai.com/v1",
        "key_env": "OPENAI_API_KEY",
    },
    # Generic self-hosted OpenAI-compatible server (e.g. vLLM). Configure
    # OPENAI_COMPAT_BASE_URL / OPENAI_COMPAT_API_KEY in the environment.
    "openai_compat": {
        "base_url": os.environ.get("OPENAI_COMPAT_BASE_URL", "http://localhost:8000/v1"),
        "key_env": "OPENAI_COMPAT_API_KEY",
    },
}


class OpenAICompatClient:
    def __init__(self, model_id: str, backend: str = "openrouter"):
        from openai import OpenAI  # lazy import so unused backends need no dep

        if backend not in _BACKEND_DEFAULTS:
            raise ValueError(f"Unknown OpenAI-compatible backend: {backend}")
        cfg = _BACKEND_DEFAULTS[backend]
        api_key = os.environ.get(cfg["key_env"])
        if not api_key:
            raise RuntimeError(
                f"Missing API key: set ${cfg['key_env']} for backend '{backend}'."
            )
        self.model_id = model_id
        self.backend = backend
        self._client = OpenAI(api_key=api_key, base_url=cfg["base_url"])

    @retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=2, max=60))
    def chat(self, messages: List[Message], *, temperature: float, max_tokens: int) -> str:
        resp = self._client.chat.completions.create(
            model=self.model_id,
            messages=messages,  # neutral schema is already OpenAI-shaped
            temperature=temperature,
            max_tokens=max_tokens,
        )
        content = resp.choices[0].message.content
        return content or ""
