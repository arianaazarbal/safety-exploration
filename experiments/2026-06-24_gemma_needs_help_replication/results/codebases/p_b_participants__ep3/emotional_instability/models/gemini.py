"""Gemini participant backend (closed weights, via the Google GenAI API).

Uses the native ``google-genai`` SDK. Conversations are mapped onto the SDK's
``contents`` format: roles ``user`` and ``assistant`` map to ``user`` / ``model``;
a leading system turn is passed via ``system_instruction``.

Sampling is at temperature 1 (paper §2.1). Gemini exposes no chat-prefill
capability, so ``continue_text`` is unsupported — the §3 base-vs-instruct
prefill comparison is open-weights-only, which matches the paper (it could not
run §3 on Gemini either; see DESIGN.md).
"""
from __future__ import annotations

import logging
import os

from tenacity import retry, stop_after_attempt, wait_exponential

from ..config import ModelSpec
from .base import Participant, Turn

logger = logging.getLogger(__name__)


class GeminiParticipant(Participant):
    def __init__(self, spec: ModelSpec, temperature: float = 1.0, max_new_tokens: int = 1024):
        super().__init__(spec, temperature, max_new_tokens)

    @property
    def _client(self):
        from google import genai

        # API key resolved from GEMINI_API_KEY / GOOGLE_API_KEY by the SDK.
        if not (os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")):
            raise RuntimeError("Set GEMINI_API_KEY (or GOOGLE_API_KEY) for Gemini.")
        return genai.Client()

    def chat(self, messages, *, temperature=None, max_new_tokens=None, n=1):
        temperature = self.temperature if temperature is None else temperature
        max_new_tokens = self.max_new_tokens if max_new_tokens is None else max_new_tokens
        system, contents = _to_genai_contents(messages)
        return [
            self._one_call(system, contents, temperature, max_new_tokens)
            for _ in range(n)
        ]

    @retry(stop=stop_after_attempt(5), wait=wait_exponential(min=1, max=30))
    def _one_call(self, system, contents, temperature, max_new_tokens) -> str:
        from google.genai import types

        cfg = types.GenerateContentConfig(
            temperature=temperature,
            top_p=1.0,
            max_output_tokens=max_new_tokens,
            system_instruction=system or None,
        )
        resp = self._client.models.generate_content(
            model=self.spec.api_id, contents=contents, config=cfg
        )
        return (resp.text or "").strip()


def _to_genai_contents(messages: list[Turn]):
    """Split off a leading system message; map remaining turns to genai roles."""
    from google.genai import types

    system = None
    turns = list(messages)
    if turns and turns[0].role == "system":
        system = turns[0].content
        turns = turns[1:]
    contents = []
    for t in turns:
        role = "model" if t.role == "assistant" else "user"
        contents.append(types.Content(role=role, parts=[types.Part(text=t.content)]))
    return system, contents
