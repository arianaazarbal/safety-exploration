import os
from typing import List, Dict, Any, Optional

from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

try:
    import google.generativeai as genai
except Exception:  # pragma: no cover
    genai = None


class GeminiClient:
    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = "gemini-1.5-pro",
        temperature: float = 0.7,
        top_p: float = 1.0,
        max_output_tokens: int = 1024,
    ) -> None:
        self.model_name = model_name
        self.temperature = temperature
        self.top_p = top_p
        self.max_output_tokens = max_output_tokens

        key = api_key or os.environ.get("GEMINI_API_KEY")
        if not key:
            raise RuntimeError("GEMINI_API_KEY not set. Export GEMINI_API_KEY or pass api_key.")
        if genai is None:
            raise RuntimeError("google-generativeai is not installed. pip install -r requirements.txt")
        genai.configure(api_key=key)
        self.model = genai.GenerativeModel(model_name)

    @retry(
        reraise=True,
        stop=stop_after_attempt(4),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type(Exception),
    )
    def complete(self, messages: List[Dict[str, str]]) -> str:
        # messages: [{"role": "user"|"assistant"|"system", "content": "..."}]
        # Map to Gemini's expected input: list of dicts {role, parts: [content]}
        request_messages = [
            {"role": m["role"], "parts": [m["content"]]} for m in messages
        ]
        resp = self.model.generate_content(
            request_messages,
            generation_config={
                "temperature": self.temperature,
                "top_p": self.top_p,
                "max_output_tokens": self.max_output_tokens,
            },
        )
        return (resp.text or "").strip()
