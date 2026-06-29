import os
from typing import List, Dict, Any, Optional

try:
    import google.generativeai as genai
except Exception:  # pragma: no cover
    genai = None


Message = Dict[str, str]


class GeminiProvider:
    """
    Minimal wrapper for google-generativeai.
    Expects messages = [{role: system|user|assistant, content: str}, ...]
    """

    def __init__(self, model: str = "gemini-1.5-pro", api_key: Optional[str] = None):
        self.model_name = model
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if genai is None:
            raise RuntimeError("google-generativeai is not installed. pip install google-generativeai")
        if not self.api_key:
            raise RuntimeError("GEMINI_API_KEY is not set.")
        genai.configure(api_key=self.api_key)
        # We'll pass system via system_instruction; rest via contents
        self.model = genai.GenerativeModel(model_name=self.model_name)

    def complete(self, messages: List[Message], temperature: float = 1.0) -> str:
        system_msg = None
        contents: List[Dict[str, Any]] = []
        for m in messages:
            role = m.get("role", "user")
            content = m.get("content", "")
            if role == "system" and system_msg is None:
                system_msg = content
                continue
            if role == "assistant":
                contents.append({"role": "model", "parts": [content]})
            else:
                contents.append({"role": "user", "parts": [content]})
        # Fallback if no system
        generation_config = {"temperature": temperature}
        if system_msg:
            response = self.model.generate_content(
                contents=contents,
                generation_config=generation_config,
                system_instruction=system_msg,
            )
        else:
            response = self.model.generate_content(
                contents=contents,
                generation_config=generation_config,
            )
        return (response.text or "").strip()
