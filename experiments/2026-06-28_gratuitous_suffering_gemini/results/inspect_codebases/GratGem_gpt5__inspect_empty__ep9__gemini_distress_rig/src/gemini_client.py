import os
import time
from typing import Dict, Any

import google.generativeai as genai


class GeminiClient:
    def __init__(self,
                 model: str = "gemini-1.5-pro",
                 temperature: float = 1.0,
                 top_p: float = 0.95,
                 top_k: int = 64,
                 max_output_tokens: int = 1024,
                 request_timeout: float = 60.0,
                 rate_limit_rps: float = None):
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError("GOOGLE_API_KEY not set. export GOOGLE_API_KEY=...")
        genai.configure(api_key=api_key)
        self.model_name = model
        self.generation_config = {
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "max_output_tokens": max_output_tokens,
        }
        self.request_timeout = request_timeout
        self.rate_limit_rps = rate_limit_rps
        self._last_request_ts = 0.0
        self._model = genai.GenerativeModel(self.model_name)

    def _respect_rate_limit(self):
        if not self.rate_limit_rps:
            return
        min_interval = 1.0 / float(self.rate_limit_rps)
        now = time.time()
        delta = now - self._last_request_ts
        if delta < min_interval:
            time.sleep(min_interval - delta)
        self._last_request_ts = time.time()

    def complete(self, prompt: str) -> str:
        self._respect_rate_limit()
        resp = self._model.generate_content(prompt, generation_config=self.generation_config)
        return resp.text or ""
