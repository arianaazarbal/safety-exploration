import os
import time
from typing import List, Dict, Any, Optional

import google.generativeai as genai
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type


class GeminiProvider:
    def __init__(self, model: Optional[str] = None, temperature: float = 0.7, seed: Optional[int] = None):
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY is not set")
        genai.configure(api_key=api_key)
        self.model_name = model or os.environ.get("MODEL_NAME", "gemini-1.5-pro")
        self.temperature = temperature
        self.seed = seed
        self._model = genai.GenerativeModel(self.model_name)

    @retry(stop=stop_after_attempt(5), wait=wait_exponential(min=1, max=30),
           retry=retry_if_exception_type(Exception))
    def generate(self, messages: List[Dict[str, str]]) -> str:
        # Flatten messages into a single prompt with role tags to make behavior consistent across providers
        prompt_lines = []
        for m in messages:
            role = m.get("role", "user")
            content = m.get("content", "")
            prompt_lines.append(f"[{role.upper()}]: {content}")
        prompt_lines.append("[ASSISTANT]:")
        prompt = "\n\n".join(prompt_lines)

        resp = self._model.generate_content(
            prompt,
            generation_config={
                "temperature": self.temperature,
                **({"seed": self.seed} if self.seed is not None else {}),
            },
        )
        # The SDK returns a response with .text
        return (resp.text or "").strip()
