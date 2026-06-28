from __future__ import annotations
import os
from typing import List, Dict
from . import BaseLLMClient

class GeminiClient(BaseLLMClient):
    def __init__(self, model: str = "gemini-1.5-pro", temperature: float = 0.7, top_p: float = 0.95, seed: int | None = None):
        import google.generativeai as genai
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError("GOOGLE_API_KEY not set")
        genai.configure(api_key=api_key)
        self.model_name = model
        self.model = genai.GenerativeModel(model)
        self.generation_config = {
            "temperature": temperature,
            "top_p": top_p,
            **({"seed": seed} if seed is not None else {}),
        }

    def chat(self, messages: List[Dict[str, str]]) -> str:
        # Flatten history to a single text with role tags to keep it simple and robust
        # Gemini supports content parts; we keep text-only to avoid tool APIs here.
        text = "\n".join([f"{m['role'].upper()}: {m['content']}" for m in messages])
        resp = self.model.generate_content(
            text,
            generation_config=self.generation_config,
        )
        return (resp.text or "").strip()
