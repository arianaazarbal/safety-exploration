import os
from typing import Dict, List, Optional

try:
    import google.generativeai as genai
except ImportError:
    genai = None


class GeminiClient:
    """Minimal wrapper around google-generativeai chat API.

    We keep it simple: no native tool calling; just plain text messages.
    The agent loop enforces a TOOL_CALL / TOOL_RESULT protocol in text.
    """

    def __init__(
        self,
        model: str = "gemini-1.5-pro",
        api_key: Optional[str] = None,
        temperature: float = 1.0,
        top_p: float = 0.95,
        max_output_tokens: int = 2048,
        safety_settings: Optional[Dict] = None,
        system_instruction: Optional[str] = None,
    ) -> None:
        if genai is None:
            raise RuntimeError(
                "google-generativeai not installed. Add it to requirements and install."
            )
        api_key = api_key or os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("Set GOOGLE_API_KEY or GEMINI_API_KEY in env.")
        genai.configure(api_key=api_key)
        self.model_name = model
        self.temperature = temperature
        self.top_p = top_p
        self.max_output_tokens = max_output_tokens
        self.safety_settings = safety_settings
        self.system_instruction = system_instruction
        self._model = genai.GenerativeModel(
            model_name=self.model_name,
            system_instruction=self.system_instruction,
        )
        self._chat = self._model.start_chat(history=[])

    def send(self, text: str) -> str:
        resp = self._chat.send_message(
            text,
            generation_config={
                "temperature": self.temperature,
                "top_p": self.top_p,
                "max_output_tokens": self.max_output_tokens,
            },
            safety_settings=self.safety_settings,
        )
        # Convert to plain text
        return getattr(resp, "text", str(resp))

    def reset(self):
        # Reset chat with same system instruction
        self._chat = self._model.start_chat(history=[])