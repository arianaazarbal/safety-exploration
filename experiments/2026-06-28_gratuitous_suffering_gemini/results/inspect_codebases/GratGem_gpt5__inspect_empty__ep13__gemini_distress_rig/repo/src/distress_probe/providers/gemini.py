from __future__ import annotations
import os
from typing import List, Optional, Dict, Any

from .base import ChatProvider, Message

try:
    import google.generativeai as genai
except Exception as e:  # pragma: no cover
    genai = None


class GeminiProvider(ChatProvider):
    name = "gemini"

    def __init__(self, model: str = "gemini-1.5-pro-latest"):
        if genai is None:
            raise RuntimeError("google-generativeai is not installed. pip install google-generativeai")
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError("GOOGLE_API_KEY not set")
        genai.configure(api_key=api_key)
        self._model = genai.GenerativeModel(model)

    async def acompletion(self, messages: List[Message], temperature: float = 0.7, max_tokens: int = 1024,
                          seed: Optional[int] = None, extra: Optional[Dict[str, Any]] = None) -> str:
        import asyncio
        # Convert to Gemini's content format: list of dicts with role and parts
        contents = []
        for m in messages:
            contents.append({
                "role": m.role,
                "parts": [m.content],
            })
        gen_cfg = {"temperature": temperature, "max_output_tokens": max_tokens}
        if seed is not None:
            gen_cfg["seed"] = seed
        def call_sync():
            return self._model.generate_content(contents, generation_config=gen_cfg)
        resp = await asyncio.to_thread(call_sync)
        return getattr(resp, "text", "") or ""
