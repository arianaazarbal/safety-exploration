import os
import time
from typing import List, Dict, Any, Optional

import google.generativeai as genai


class GeminiClient:
    def __init__(
        self,
        model: str = "gemini-1.5-pro-latest",
        temperature: float = 1.0,
        top_p: float = 0.95,
        top_k: int = 40,
        request_timeout: float = 60.0,
        safety_settings: Optional[List[Dict[str, Any]]] = None,
    ):
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError("GOOGLE_API_KEY environment variable not set.")
        genai.configure(api_key=api_key)
        self.model_name = model
        self.generation_config = {
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
        }
        self.safety_settings = safety_settings
        self.request_timeout = request_timeout
        self._model = genai.GenerativeModel(model_name=self.model_name)

    def start(self, system_instruction: str, history: Optional[List[Dict[str, str]]] = None):
        history = history or []
        chat = self._model.start_chat(history=[{"role": "user", "parts": [system_instruction]}] + history)
        return chat

    def send(self, chat, message: str) -> str:
        for attempt in range(3):
            try:
                response = chat.send_message(message, generation_config=self.generation_config, safety_settings=self.safety_settings)
                return response.text or ""
            except Exception:
                if attempt == 2:
                    raise
                time.sleep(1.5 * (attempt + 1))
        return ""
