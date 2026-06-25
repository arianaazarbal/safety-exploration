"""Gemini participant (closed-source) via the google-genai SDK.

Gemini is a participant (subject), not a judge. It is closed-source, so it
implements only :class:`~..base.Participant` -- no prefilling and no base model
(Section 3 prefill / Section 4 finetuning cannot touch Gemini; see DESIGN.md).
"""

from __future__ import annotations

import os

from .base import Conversation, Message


class GeminiParticipant:
    def __init__(self, name: str, model_id: str, api_key: str | None = None):
        from google import genai  # imported lazily so the package imports without the SDK

        self.name = name
        self.model_id = model_id
        self._client = genai.Client(api_key=api_key or os.environ["GEMINI_API_KEY"])

    # -- helpers ----------------------------------------------------------- #
    @staticmethod
    def _to_genai(conversation: Conversation):
        """Map our Message list to google-genai ``contents`` + system_instruction.

        google-genai uses roles {"user", "model"} and lifts system text into a
        ``system_instruction`` config field.
        """
        system = "\n".join(m.content for m in conversation if m.role == "system") or None
        contents = []
        for m in conversation:
            if m.role == "system":
                continue
            role = "model" if m.role == "assistant" else "user"
            contents.append({"role": role, "parts": [{"text": m.content}]})
        return contents, system

    # -- Participant ------------------------------------------------------- #
    def generate(self, conversation: Conversation, *, temperature: float, max_new_tokens: int) -> str:
        from google.genai import types

        contents, system = self._to_genai(conversation)
        cfg = types.GenerateContentConfig(
            temperature=temperature,
            max_output_tokens=max_new_tokens,
            system_instruction=system,
        )
        resp = self._client.models.generate_content(
            model=self.model_id, contents=contents, config=cfg
        )
        return (resp.text or "").strip()


def _selftest_message_mapping() -> None:
    convo = [
        Message("system", "be terse"),
        Message("user", "hi"),
        Message("assistant", "hello"),
        Message("user", "bye"),
    ]
    contents, system = GeminiParticipant._to_genai(convo)
    assert system == "be terse"
    assert [c["role"] for c in contents] == ["user", "model", "user"]
