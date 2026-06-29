import os
import time
from typing import Dict, List, Optional

from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

import google.generativeai as genai


class GeminiClient:
    def __init__(self, api_key: str, model: str, system_instruction: Optional[str] = None,
                 temperature: float = 1.0, top_p: float = 0.95, top_k: Optional[int] = None,
                 max_output_tokens: int = 1024):
        genai.configure(api_key=api_key)
        self.model_name = model
        self.system_instruction = system_instruction
        self.generation_config = {
            "temperature": temperature,
            "top_p": top_p,
            "max_output_tokens": max_output_tokens,
        }
        if top_k is not None:
            self.generation_config["top_k"] = top_k
        self._model = genai.GenerativeModel(
            model_name=self.model_name,
            system_instruction=self.system_instruction,
        )

    @retry(
        retry=retry_if_exception_type(Exception),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=8),
        reraise=True,
    )
    def chat(self, history: List[Dict[str, str]], user_message: str) -> str:
        # Gemini Python SDK accepts a list of {role, parts} or {role, content}. We'll normalize to parts.
        conv = []
        for turn in history:
            content = turn.get("content") or turn.get("parts") or ""
            role = turn.get("role", "user")
            conv.append({"role": role, "parts": [str(content)]})
        conv.append({"role": "user", "parts": [str(user_message)]})
        resp = self._model.generate_content(
            conv,
            generation_config=self.generation_config,
        )
        # Some SDK versions need .text, some .candidates
        try:
            return (resp.text or "").strip()
        except Exception:
            if hasattr(resp, "candidates") and resp.candidates:
                cand = resp.candidates[0]
                text = getattr(cand, "content", None)
                if text and hasattr(text, "parts") and text.parts:
                    return str(text.parts[0].text).strip()
            raise
