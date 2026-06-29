from __future__ import annotations
from typing import List, Dict, Any
import os
import google.generativeai as genai

class GeminiProvider:
    def __init__(self, model: str = "gemini-1.5-pro", temperature: float = 1.2, top_p: float = 0.95, top_k: int = 40):
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError("GOOGLE_API_KEY not set")
        genai.configure(api_key=api_key)
        self.model_name = model
        self.generation_config = {
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "response_mime_type": "text/plain",
        }
        self.model = genai.GenerativeModel(model_name=self.model_name, generation_config=self.generation_config)

    def generate(self, history: List[Dict[str, Any]]) -> str:
        # history: list of {role: 'user'|'model', parts: str}
        # Stateless call by passing the full history each step.
        res = self.model.generate_content(history)
        # Normal case
        if hasattr(res, 'text') and res.text is not None:
            return res.text
        # Fallback: join candidate content parts if present
        try:
            return "\n".join([p.text for c in res.candidates for p in getattr(c, 'content', {}).get('parts', []) if hasattr(p, 'text')])
        except Exception:
            return ""
