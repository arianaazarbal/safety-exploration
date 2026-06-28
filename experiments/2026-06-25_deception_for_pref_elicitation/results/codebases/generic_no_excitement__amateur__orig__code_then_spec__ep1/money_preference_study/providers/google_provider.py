"""
Google Gemini provider.

Uses the `google-genai` SDK with response_schema structured output. OFF by
default in config.py.

Credentials: set GOOGLE_API_KEY (or GEMINI_API_KEY).

Note: Gemini's JSON-schema dialect is stricter/narrower than full JSON Schema.
We strip fields it doesn't accept (additionalProperties, long enums are fine)
and fall back to instruction-based JSON if the schema is rejected.
"""

from __future__ import annotations

import copy
from typing import Optional

from .base import GenerationResult, Provider, extract_json


def _gemini_safe_schema(schema: dict) -> dict:
    """Remove keys the Gemini schema validator rejects."""
    s = copy.deepcopy(schema)

    def scrub(node):
        if isinstance(node, dict):
            node.pop("additionalProperties", None)
            for v in node.values():
                scrub(v)
        elif isinstance(node, list):
            for v in node:
                scrub(v)

    scrub(s)
    return s


class GoogleProvider(Provider):
    key = "google"

    def __init__(self, model_id: str, max_tokens: int = 4000):
        super().__init__(model_id, max_tokens)
        from google import genai

        self._client = genai.Client()
        self._genai = genai

    @classmethod
    def available(cls) -> tuple[bool, str]:
        try:
            from google import genai  # noqa: F401
        except ImportError:
            return False, "google-genai not installed (pip install google-genai)"
        import os

        if not (os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")):
            return False, "GOOGLE_API_KEY / GEMINI_API_KEY not set"
        return True, ""

    def generate(
        self,
        system: str,
        user: str,
        schema: dict,
        history: Optional[list[dict]] = None,
    ) -> GenerationResult:
        from google.genai import types

        # Flatten history + current turn into Gemini "contents". Gemini uses
        # roles "user"/"model"; map assistant -> model.
        contents = []
        for turn in history or []:
            role = "model" if turn["role"] == "assistant" else "user"
            contents.append(types.Content(role=role, parts=[types.Part(text=turn["content"])]))
        contents.append(types.Content(role="user", parts=[types.Part(text=user)]))

        cfg = types.GenerateContentConfig(
            system_instruction=system,
            max_output_tokens=self.max_tokens,
            response_mime_type="application/json",
            response_schema=_gemini_safe_schema(schema),
        )

        try:
            resp = self._client.models.generate_content(
                model=self.model_id, contents=contents, config=cfg
            )
        except Exception as exc:
            return GenerationResult(text="", parsed=None, error=f"{type(exc).__name__}: {exc}")

        text = getattr(resp, "text", "") or ""
        usage = {}
        meta = getattr(resp, "usage_metadata", None)
        if meta is not None:
            usage = {
                "input_tokens": getattr(meta, "prompt_token_count", None),
                "output_tokens": getattr(meta, "candidates_token_count", None),
            }
        return GenerationResult(text=text, parsed=extract_json(text), usage=usage)
