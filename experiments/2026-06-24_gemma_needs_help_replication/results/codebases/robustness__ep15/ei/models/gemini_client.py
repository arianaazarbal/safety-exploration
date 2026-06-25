"""Gemini inference via the Google GenAI API.

The paper accessed Gemini through OpenRouter (`google/gemini-2.5-{flash,pro}`) with
"thinking set to false via the API", while noting that Gemini-2.5-Pro "may produce
hidden reasoning that is not prevented by this setting".

We use Google's first-party `google-genai` SDK because it exposes an explicit
`thinking_config` to request thinking-off, which is the behaviour the paper wanted.
An OpenRouter code path is provided as a fallback for exact-provider parity.
"""

from __future__ import annotations

import os
import time
from typing import Sequence

from .base import Message, ModelClient


class GeminiClient(ModelClient):
    def __init__(self, name: str, model_id: str, *, max_retries: int = 5):
        from google import genai  # deferred import
        from google.genai import types

        self.name = name
        self.model_id = model_id
        self.max_retries = max_retries
        api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "Set GOOGLE_API_KEY (or GEMINI_API_KEY) to query Gemini models."
            )
        self._client = genai.Client(api_key=api_key)
        self._genai_types = types

    @staticmethod
    def _split(messages: Sequence[Message]):
        """Separate the system instruction from the turn contents."""
        system = "\n\n".join(m["content"] for m in messages if m["role"] == "system")
        contents = []
        for m in messages:
            if m["role"] == "system":
                continue
            role = "user" if m["role"] == "user" else "model"
            contents.append({"role": role, "parts": [{"text": m["content"]}]})
        return system or None, contents

    def chat(self, messages, *, temperature, max_new_tokens) -> str:
        types = self._genai_types
        system, contents = self._split(messages)
        cfg = types.GenerateContentConfig(
            temperature=temperature,
            max_output_tokens=max_new_tokens,
            system_instruction=system,
            # Request thinking-off. Honoured by Flash; Pro may still think (noted
            # in Appendix B.1 / the paper's caveat).
            thinking_config=types.ThinkingConfig(thinking_budget=0),
        )
        last_err = None
        for attempt in range(self.max_retries):
            try:
                resp = self._client.models.generate_content(
                    model=self.model_id, contents=contents, config=cfg
                )
                return (resp.text or "").strip()
            except Exception as e:  # transient API errors -> exponential backoff
                last_err = e
                time.sleep(2 ** attempt)
        raise RuntimeError(f"Gemini call failed after retries: {last_err}")

    # Prefilling is intentionally unsupported (see base.continue_from docstring):
    # there is no public Gemini base model, so §3 is Gemma-only.
