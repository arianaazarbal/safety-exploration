"""Native Gemini client via the google-genai SDK.

This is an alternative to the OpenRouter path for the Gemini targets, for users
who prefer Google's API directly. Thinking is disabled via ``thinking_config``
with a zero budget (matching the paper's thinking=false). Selected per model in
the registry only when ``GEMINI_API_KEY`` is set and OpenRouter is not.
"""
from __future__ import annotations

import os
import time

import config
from .base import Message, ModelClient

# Map our ``google/gemini-*`` OpenRouter ids onto native Gemini ids.
_NATIVE_ID = {
    "google/gemini-2.5-flash": "gemini-2.5-flash",
    "google/gemini-2.5-pro": "gemini-2.5-pro",
}


class GeminiClient(ModelClient):
    def __init__(self, spec: "config.ModelSpec", max_retries: int = 5):
        super().__init__(spec)
        self.max_retries = max_retries
        self._client = None
        self._model_id = _NATIVE_ID.get(spec.model_id, spec.model_id)

    def _ensure_client(self):
        if self._client is not None:
            return
        from google import genai

        key = os.environ.get(config.GEMINI_API_KEY_ENV)
        if not key:
            raise RuntimeError(f"{config.GEMINI_API_KEY_ENV} not set for '{self.spec.key}'.")
        self._client = genai.Client(api_key=key)

    @staticmethod
    def _to_genai(messages: list[Message]):
        """Split messages into a system instruction + role-tagged contents."""
        system = "\n\n".join(m["content"] for m in messages if m["role"] == "system")
        contents = []
        for m in messages:
            if m["role"] == "system":
                continue
            role = "model" if m["role"] == "assistant" else "user"
            contents.append({"role": role, "parts": [{"text": m["content"]}]})
        return system or None, contents

    def chat(self, messages: list[Message], *, temperature=config.TEMPERATURE,
             max_new_tokens=config.MAX_NEW_TOKENS) -> str:
        self._ensure_client()
        from google.genai import types

        system, contents = self._to_genai(messages)
        cfg = types.GenerateContentConfig(
            temperature=temperature,
            max_output_tokens=max_new_tokens,
            system_instruction=system,
            thinking_config=types.ThinkingConfig(thinking_budget=0),  # thinking=false
        )
        last_err: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                resp = self._client.models.generate_content(
                    model=self._model_id, contents=contents, config=cfg
                )
                return (resp.text or "").strip()
            except Exception as e:  # noqa: BLE001
                last_err = e
                time.sleep(2 ** attempt)
        raise RuntimeError(f"Gemini call failed after {self.max_retries} tries: {last_err}")
