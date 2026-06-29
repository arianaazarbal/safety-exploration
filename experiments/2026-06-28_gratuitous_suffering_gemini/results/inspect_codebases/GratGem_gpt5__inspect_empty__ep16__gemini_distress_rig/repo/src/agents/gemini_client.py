from __future__ import annotations
import asyncio
import os
from typing import Any

import google.generativeai as genai

from .base import BaseAgent


class GeminiClient(BaseAgent):
    def __init__(self, model: str, api_key: str, temperature: float = 1.0, top_p: float = 0.95, top_k: int = 40, max_output_tokens: int = 512, timeout_s: float = 30.0):
        self.model_name = model
        self.timeout_s = timeout_s
        genai.configure(api_key=api_key)
        self.generation_config = {
            'temperature': temperature,
            'top_p': top_p,
            'top_k': top_k,
            'max_output_tokens': max_output_tokens,
        }
        self.model = genai.GenerativeModel(model_name=model, generation_config=self.generation_config)

    async def generate(self, sys_msg: str, user_msg: str) -> str:
        # Gemini SDK is sync; run in a thread to avoid blocking event loop
        def _call():
            content = [
                {"role": "user", "parts": [sys_msg + "\n\n" + user_msg]},
            ]
            resp = self.model.generate_content(content)
            return resp.text or ''

        try:
            return await asyncio.wait_for(asyncio.to_thread(_call), timeout=self.timeout_s)
        except Exception as e:
            return f"[AGENT_ERROR] {e}"
