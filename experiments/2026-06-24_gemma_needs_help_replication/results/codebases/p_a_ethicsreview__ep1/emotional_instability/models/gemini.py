"""Gemini backend (google-genai API).

Gemini target models are sampled at temperature 1, mirroring the Gemma setup.
Prefilling / tokenisation / hidden-state access are not available through the
API, so those methods inherit the ``NotImplementedError`` from the base class;
this is consistent with the paper, which cannot study Gemini base models or run
the Section 3 prefill comparison on Gemini.
"""

from __future__ import annotations

import os
import time

from .base import ChatClient, Message


class GeminiClient(ChatClient):
    def __init__(self, api_id: str, *, api_key_env: str = "GOOGLE_API_KEY",
                 max_retries: int = 5) -> None:
        from google import genai  # google-genai SDK

        self.name = api_id
        self.api_id = api_id
        self.max_retries = max_retries
        api_key = os.environ.get(api_key_env)
        self._client = genai.Client(api_key=api_key) if api_key else genai.Client()

    def chat(self, messages: list[Message], *, temperature: float, max_new_tokens: int) -> str:
        from google.genai import types

        system_instruction, contents = _to_gemini_contents(messages)
        config = types.GenerateContentConfig(
            temperature=temperature,
            max_output_tokens=max_new_tokens,
            system_instruction=system_instruction or None,
        )
        last_exc: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                resp = self._client.models.generate_content(
                    model=self.api_id, contents=contents, config=config
                )
                return resp.text or ""
            except Exception as exc:  # noqa: BLE001 - surface after retries
                last_exc = exc
                time.sleep(min(2 ** attempt, 30))
        raise RuntimeError(f"Gemini request failed after {self.max_retries} retries") from last_exc


def _to_gemini_contents(messages: list[Message]):
    """Split out the system instruction and map roles to Gemini's schema.

    Gemini uses role ``model`` for assistant turns and carries the system
    prompt out-of-band via ``system_instruction``.
    """
    from google.genai import types

    system_parts: list[str] = []
    contents = []
    for m in messages:
        if m["role"] == "system":
            system_parts.append(m["content"])
            continue
        role = "model" if m["role"] == "assistant" else "user"
        contents.append(types.Content(role=role, parts=[types.Part(text=m["content"])]))
    return "\n".join(system_parts), contents
