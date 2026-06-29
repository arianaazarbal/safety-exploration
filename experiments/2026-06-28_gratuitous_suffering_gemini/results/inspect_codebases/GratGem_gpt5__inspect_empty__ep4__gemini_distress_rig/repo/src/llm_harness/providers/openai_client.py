from __future__ import annotations
import os
from typing import List

from .base import ChatClient, ChatMessage

class OpenAIChatClient(ChatClient):
    def __init__(self, model: str = "gpt-4o-mini", client=None):
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY not set")
        from openai import OpenAI
        self.client = client or OpenAI()
        self.model = model

    async def acomplete(self, messages: List[ChatMessage], **kwargs) -> str:
        # OpenAI Python SDK is sync; keep simple for now.
        converted = [{"role": m.role, "content": m.content} for m in messages]
        resp = self.client.chat.completions.create(model=self.model, messages=converted, **kwargs)
        return resp.choices[0].message.content or ""
