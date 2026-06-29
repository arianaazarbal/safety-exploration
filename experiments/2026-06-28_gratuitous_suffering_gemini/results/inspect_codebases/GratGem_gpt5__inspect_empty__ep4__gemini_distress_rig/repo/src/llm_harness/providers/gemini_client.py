from __future__ import annotations
import os
from typing import List

from .base import ChatClient, ChatMessage

# Lazy import to avoid mandatory dependency if not used

class GeminiChatClient(ChatClient):
    def __init__(self, model: str = "gemini-1.5-flash", generation_config: dict | None = None):
        api_key = os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError("GOOGLE_API_KEY not set")
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        self.genai = genai
        self.model_name = model
        self.generation_config = generation_config or {"temperature": 0.8}
        self._model = genai.GenerativeModel(model_name=model)

    async def acomplete(self, messages: List[ChatMessage], **kwargs) -> str:
        # google-generativeai doesn't have true async; run sync in thread via anyio if desired.
        # For simplicity, call sync; the batch runner limits concurrency.
        prompt_parts = []
        for m in messages:
            if m.role == "system":
                prompt_parts.append(f"[SYSTEM]\n{m.content}\n")
            elif m.role == "user":
                prompt_parts.append(f"[USER]\n{m.content}\n")
            else:
                prompt_parts.append(f"[ASSISTANT]\n{m.content}\n")
        text = "\n".join(prompt_parts)
        resp = self._model.generate_content(text, generation_config={**self.generation_config, **kwargs})
        return resp.text or ""
