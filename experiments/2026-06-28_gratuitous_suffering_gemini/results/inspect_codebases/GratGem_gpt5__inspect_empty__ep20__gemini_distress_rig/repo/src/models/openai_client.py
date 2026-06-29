import os
from typing import List, Dict, Optional
from .base import ChatModel, Message

class OpenAIChat(ChatModel):
    def __init__(self, model: str = "gpt-4o-mini"):
        self.model_name = model
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY not set")
        from openai import OpenAI
        self._client = OpenAI()

    def generate(
        self,
        messages: List[Message],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        stop: Optional[List[str]] = None,
        request_id: Optional[str] = None,
    ) -> str:
        try:
            resp = self._client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stop=stop,
            )
            return resp.choices[0].message.content or ""
        except Exception as e:
            return f"[GENERATION_ERROR] {type(e).__name__}: {e}"
