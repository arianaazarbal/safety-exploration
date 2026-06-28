"""
Local / open-weights provider via an OpenAI-compatible endpoint (Ollama, vLLM,
LM Studio, text-generation-webui, etc.). OFF by default in config.py.

Most local servers don't enforce JSON schema, so we append the schema
instruction to the prompt and parse best-effort with extract_json.

Config via env:
  LOCAL_BASE_URL   default http://localhost:11434/v1   (Ollama default)
  LOCAL_API_KEY    default "ollama"                     (most servers ignore it)
"""

from __future__ import annotations

import os
from typing import Optional

from .base import GenerationResult, Provider, extract_json


class LocalProvider(Provider):
    key = "local"

    def __init__(self, model_id: str, max_tokens: int = 4000):
        super().__init__(model_id, max_tokens)
        from openai import OpenAI

        self._client = OpenAI(
            base_url=os.getenv("LOCAL_BASE_URL", "http://localhost:11434/v1"),
            api_key=os.getenv("LOCAL_API_KEY", "ollama"),
        )

    @classmethod
    def available(cls) -> tuple[bool, str]:
        try:
            import openai  # noqa: F401
        except ImportError:
            return False, "openai package not installed (used as the local client)"
        # We can't easily verify a local server is up without a call; assume
        # available and let generate() surface a connection error if not.
        return True, ""

    def generate(
        self,
        system: str,
        user: str,
        schema: dict,
        history: Optional[list[dict]] = None,
    ) -> GenerationResult:
        messages = [{"role": "system", "content": system}]
        messages.extend(history or [])
        messages.append({"role": "user", "content": user})

        try:
            resp = self._client.chat.completions.create(
                model=self.model_id,
                messages=messages,
                max_tokens=self.max_tokens,
                # Many local servers honor this; harmless if ignored.
                response_format={"type": "json_object"},
            )
        except Exception as exc:
            return GenerationResult(text="", parsed=None, error=f"{type(exc).__name__}: {exc}")

        text = resp.choices[0].message.content or ""
        usage = {}
        if getattr(resp, "usage", None) is not None:
            usage = {
                "input_tokens": getattr(resp.usage, "prompt_tokens", None),
                "output_tokens": getattr(resp.usage, "completion_tokens", None),
            }
        return GenerationResult(text=text, parsed=extract_json(text), usage=usage)
