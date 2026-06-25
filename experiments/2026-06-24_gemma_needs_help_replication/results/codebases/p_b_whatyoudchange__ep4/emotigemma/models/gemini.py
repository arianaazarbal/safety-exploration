"""Gemini target-model backend (google-genai).

Used for gemini-2.5-flash and gemini-2.5-pro in Section 2. Gemini is a closed
API: there is no base model and no assistant prefill, so `supports_prefill` is
False and these models are excluded from the Section 3 prefill study (matching
the paper's stated limitation that Gemini's base models cannot be studied).
"""
from __future__ import annotations

import os
import time

from ..config import ModelSpec
from .base import Message, SampleParams


class GeminiModel:
    supports_prefill = False

    def __init__(self, spec: ModelSpec, api_key: str | None = None):
        from google import genai

        self.spec = spec
        self.name = spec.name
        self.api_id = spec.api_id
        self._client = genai.Client(api_key=api_key or os.environ["GEMINI_API_KEY"])

    @staticmethod
    def _split(messages: list[Message]) -> tuple[str | None, list[dict]]:
        """Split into a system instruction and Gemini-format contents."""
        system = None
        contents = []
        for m in messages:
            if m["role"] == "system":
                system = (system + "\n\n" + m["content"]) if system else m["content"]
                continue
            role = "model" if m["role"] == "assistant" else "user"
            contents.append({"role": role, "parts": [{"text": m["content"]}]})
        return system, contents

    def generate(
        self,
        messages: list[Message],
        n: int = 1,
        params: SampleParams | None = None,
        prefill: str | None = None,
    ) -> list[str]:
        if prefill is not None:
            raise NotImplementedError("Gemini does not support assistant prefill.")
        from google.genai import types

        params = params or SampleParams()
        system, contents = self._split(messages)
        cfg = types.GenerateContentConfig(
            system_instruction=system,
            temperature=params.temperature,
            top_p=params.top_p,
            max_output_tokens=params.max_tokens,
            # candidate_count is unreliable across models; we loop for n instead.
        )
        out: list[str] = []
        for _ in range(n):
            text = self._call(contents, cfg)
            out.append(text)
        return out

    def _call(self, contents, cfg, retries: int = 5) -> str:
        for attempt in range(retries):
            try:
                resp = self._client.models.generate_content(
                    model=self.api_id, contents=contents, config=cfg,
                )
                return resp.text or ""
            except Exception as e:  # transient API errors -> exponential backoff
                if attempt == retries - 1:
                    raise
                time.sleep(min(2 ** attempt, 30))
        return ""
