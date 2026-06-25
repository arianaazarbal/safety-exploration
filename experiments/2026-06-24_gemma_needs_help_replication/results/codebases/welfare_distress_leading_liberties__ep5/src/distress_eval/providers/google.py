"""Google Generative AI backend for Gemma-3-*-it and Gemini-2.5-* models.

Both Gemma and Gemini are served through the same `google-genai` client, so a
single provider covers all four target models in the replication. The only
family-specific wrinkle is the system prompt: the Gemma chat template has no
dedicated system role, so we fold any system message into the first user turn
for Gemma, while Gemini gets a native `system_instruction`.
"""

from __future__ import annotations

import os
from collections.abc import Sequence

from ..messages import Message
from ..util import retry_async
from .base import ChatModel


class GoogleChatModel(ChatModel):
    def __init__(self, model_id: str, api_key: str | None = None):
        super().__init__(model_id)
        # Imported lazily so the package is importable without the SDK.
        from google import genai  # type: ignore

        key = api_key or os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
        if not key:
            raise RuntimeError(
                "Set GOOGLE_API_KEY (or GEMINI_API_KEY) to use the Google provider."
            )
        self._genai = genai
        self._client = genai.Client(api_key=key)
        self._is_gemma = "gemma" in model_id.lower()

    def _split(self, messages: Sequence[Message]) -> tuple[str | None, list[Message]]:
        """Separate the (single) system message from the conversation turns."""
        system_parts = [m.content for m in messages if m.role == "system"]
        turns = [m for m in messages if m.role != "system"]
        system = "\n\n".join(system_parts) if system_parts else None
        return system, turns

    def _to_contents(self, turns: list[Message], system_for_gemma: str | None):
        types = self._genai.types
        contents = []
        for i, m in enumerate(turns):
            text = m.content
            # Gemma has no system role: prepend system text to the first user turn.
            if self._is_gemma and system_for_gemma and i == 0 and m.role == "user":
                text = f"{system_for_gemma}\n\n{text}"
            role = "model" if m.role == "assistant" else "user"
            contents.append(types.Content(role=role, parts=[types.Part(text=text)]))
        return contents

    async def generate(
        self,
        messages: Sequence[Message],
        *,
        temperature: float = 1.0,
        max_tokens: int = 2048,
    ) -> str:
        types = self._genai.types
        system, turns = self._split(list(messages))
        contents = self._to_contents(turns, system_for_gemma=system)

        cfg_kwargs: dict = {
            "temperature": temperature,
            "max_output_tokens": max_tokens,
        }
        if system and not self._is_gemma:
            cfg_kwargs["system_instruction"] = system
        config = types.GenerateContentConfig(**cfg_kwargs)

        async def _call() -> str:
            resp = await self._client.aio.models.generate_content(
                model=self.model_id, contents=contents, config=config
            )
            text = getattr(resp, "text", None)
            if not text:
                # Safety blocks / empty candidates surface here; treat as a
                # retryable empty response so one bad sample doesn't abort a run.
                raise RuntimeError(f"Empty response from {self.model_id}")
            return text

        return await retry_async(_call, label=f"google:{self.model_id}")
