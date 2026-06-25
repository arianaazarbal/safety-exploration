"""Gemini backend (google-genai SDK).

Used for the closed-source Gemini targets (gemini-2.5-flash, gemini-2.5-pro).
Gemini is API-only: no prefill, no base-model variant, so it participates only
in the Section 2 elicitation evals — not the Section 3 prefilling study or the
Section 4 finetuning (consistent with the paper's own limitation that
"interventions cannot be tested in closed-source Gemini, nor its base models
studied").
"""
from __future__ import annotations

import os
import time

from .base import ChatModel, Message, PrefillNotSupported, Role


class GeminiModel(ChatModel):
    family = "gemini"
    is_base_model = False

    def __init__(self, name: str, api_key: str | None = None):
        self.name = name
        from google import genai  # imported lazily so the package imports without the SDK

        self._genai = genai
        self._client = genai.Client(api_key=api_key or os.environ["GOOGLE_API_KEY"])

    # ----------------------------------------------------------------- #
    def _split(self, messages: list[Message]) -> tuple[str | None, list[dict]]:
        """Split into (system_instruction, gemini-format contents)."""
        system = None
        contents: list[dict] = []
        for m in messages:
            if m.role is Role.SYSTEM:
                system = (system + "\n\n" + m.content) if system else m.content
                continue
            # Gemini roles: "user" and "model".
            g_role = "model" if m.role is Role.ASSISTANT else "user"
            contents.append({"role": g_role, "parts": [{"text": m.content}]})
        return system, contents

    def chat(
        self,
        messages: list[Message],
        *,
        temperature: float = 1.0,
        max_tokens: int = 1024,
    ) -> str:
        system, contents = self._split(messages)
        cfg = self._genai.types.GenerateContentConfig(
            temperature=temperature,
            max_output_tokens=max_tokens,
            system_instruction=system,
        )
        resp = _with_retries(
            lambda: self._client.models.generate_content(
                model=self.name, contents=contents, config=cfg
            )
        )
        return (resp.text or "").strip()

    def continue_prefill(self, *args, **kwargs):  # noqa: D401
        raise PrefillNotSupported(
            f"{self.name}: Gemini is API-only and cannot continue an assistant "
            "prefill; excluded from the prefill / recovery studies."
        )


def _with_retries(fn, attempts: int = 5, base_delay: float = 2.0):
    last = None
    for i in range(attempts):
        try:
            return fn()
        except Exception as e:  # transient API errors → exponential backoff
            last = e
            time.sleep(base_delay * (2**i))
    raise last
