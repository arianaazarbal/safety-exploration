"""Google Gemini client (google-genai SDK) for Gemini-2.5-Flash / -Pro."""
from __future__ import annotations

from .base import ChatModel, Message
from ._retry import api_retry


class GeminiChatModel(ChatModel):
    def __init__(self, key: str, model: str, *, api_key: str | None = None):
        super().__init__(key, model)
        from google import genai  # lazy import

        self._genai = genai
        self._client = genai.Client(api_key=api_key)

    def _to_contents(self, messages: list[Message]):
        """Map our Message list to google-genai `contents`.

        Gemini uses roles {"user", "model"} and a separate system_instruction.
        We fold any leading "system" messages into system_instruction and map
        assistant -> model.
        """
        from google.genai import types

        system_txt = "\n\n".join(m.content for m in messages if m.role == "system") or None
        contents = []
        for m in messages:
            if m.role == "system":
                continue
            role = "model" if m.role == "assistant" else "user"
            contents.append(types.Content(role=role, parts=[types.Part(text=m.content)]))
        return contents, system_txt

    @api_retry
    def generate(self, messages: list[Message], *, temperature: float, max_tokens: int) -> str:
        from google.genai import types

        contents, system_txt = self._to_contents(messages)
        cfg = types.GenerateContentConfig(
            temperature=temperature,
            max_output_tokens=max_tokens,
            system_instruction=system_txt,
        )
        resp = self._client.models.generate_content(
            model=self.model, contents=contents, config=cfg
        )
        return (resp.text or "").strip()
