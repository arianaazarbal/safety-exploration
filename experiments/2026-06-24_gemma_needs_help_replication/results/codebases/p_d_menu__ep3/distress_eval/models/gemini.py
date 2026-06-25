"""Gemini backend (google-genai SDK).

Multi-turn chat with temperature control. Gemini is closed-weight, so there is
no base model and no prefilled continuation (both raise NotImplementedError via
the base class).
"""

from __future__ import annotations

import logging
import os
import time
from typing import TYPE_CHECKING

from .base import ChatMessage, ModelClient

if TYPE_CHECKING:
    from config import ModelSpec

log = logging.getLogger(__name__)


class GeminiClient(ModelClient):
    def __init__(self, spec: "ModelSpec", api_key: str | None = None, max_retries: int = 4):
        self.spec = spec
        self._max_retries = max_retries
        self._api_key = api_key or os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
        self._client = None

    def _ensure_client(self):
        if self._client is None:
            from google import genai

            self._client = genai.Client(api_key=self._api_key)
        return self._client

    def chat(self, messages, temperature=1.0, max_new_tokens=1024, stop=None) -> str:
        from google.genai import types

        client = self._ensure_client()

        system_instruction = None
        contents = []
        for m in messages:
            if m["role"] == "system":
                system_instruction = m["content"]
                continue
            role = "user" if m["role"] == "user" else "model"
            contents.append(types.Content(role=role, parts=[types.Part(text=m["content"])]))

        cfg = types.GenerateContentConfig(
            temperature=temperature,
            max_output_tokens=max_new_tokens,
            system_instruction=system_instruction,
            stop_sequences=stop or None,
        )

        last_exc = None
        for attempt in range(self._max_retries):
            try:
                resp = client.models.generate_content(
                    model=self.spec.model_id, contents=contents, config=cfg,
                )
                return (resp.text or "").strip()
            except Exception as exc:  # pragma: no cover - network dependent
                last_exc = exc
                wait = min(2 ** attempt, 30)
                log.warning("Gemini call failed (attempt %d): %s; retrying in %ds",
                            attempt + 1, exc, wait)
                time.sleep(wait)
        raise RuntimeError(f"Gemini call failed after {self._max_retries} retries") from last_exc
