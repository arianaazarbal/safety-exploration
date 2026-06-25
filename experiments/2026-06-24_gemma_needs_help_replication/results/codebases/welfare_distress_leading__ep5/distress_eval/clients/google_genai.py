"""Google GenAI backend (Gemini 2.5 + Gemma 3 via Google AI Studio / Vertex).

Google AI Studio serves both Gemini and the open Gemma models behind the same
``generate_content`` API, so one backend covers both target families when using
Google directly instead of OpenRouter.

Note: Gemma has no system role. We prepend any system text to the first user
turn rather than passing it as a system instruction.
"""

from __future__ import annotations

import os
from typing import List

from tenacity import retry, stop_after_attempt, wait_exponential

from .base import Message, split_system


class GoogleGenAIClient:
    def __init__(self, model_id: str):
        from google import genai  # lazy import

        api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("Set $GOOGLE_API_KEY (or $GEMINI_API_KEY) for the google backend.")
        self.model_id = model_id
        self._genai = genai
        self._client = genai.Client(api_key=api_key)

    def _to_contents(self, messages: List[Message]):
        """Translate neutral messages -> google.genai Content list.

        Roles map user->user, assistant->model. A leading system message is
        folded into the first user turn (Gemma/Gemini-on-AI-Studio friendly).
        """
        types = self._genai.types
        system, rest = split_system(messages)
        contents = []
        for i, m in enumerate(rest):
            role = "model" if m["role"] == "assistant" else "user"
            text = m["content"]
            if system and i == 0 and role == "user":
                text = f"{system}\n\n{text}"
                system = None
            contents.append(types.Content(role=role, parts=[types.Part(text=text)]))
        return contents

    @retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=2, max=60))
    def chat(self, messages: List[Message], *, temperature: float, max_tokens: int) -> str:
        types = self._genai.types
        resp = self._client.models.generate_content(
            model=self.model_id,
            contents=self._to_contents(messages),
            config=types.GenerateContentConfig(
                temperature=temperature,
                max_output_tokens=max_tokens,
            ),
        )
        return resp.text or ""
