import os
from typing import List, Dict, Optional
from .base import ChatModel, Message

# Lightweight adapter over google-generativeai.
# We construct a single-turn prompt from the message list to avoid stateful chat complexity.

class GeminiChat(ChatModel):
    def __init__(self, model: str = "gemini-1.5-pro"):
        self.model_name = model
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY not set")
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        self._genai = genai
        self._model = genai.GenerativeModel(self.model_name)

    def _messages_to_content(self, messages: List[Message]) -> List[Dict]:
        # Convert role/content pairs into Google content list. Map 'assistant' -> 'model'.
        role_map = {"assistant": "model"}
        content = []
        for m in messages:
            role = m.get("role", "user")
            role = role_map.get(role, role)
            text = m.get("content", "")
            if not text:
                continue
            content.append({"role": role, "parts": [text]})
        return content

    def generate(
        self,
        messages: List[Message],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        stop: Optional[List[str]] = None,
        request_id: Optional[str] = None,
    ) -> str:
        content = self._messages_to_content(messages)
        try:
            resp = self._model.generate_content(
                content,
                generation_config={
                    "temperature": float(temperature),
                    **({"max_output_tokens": int(max_tokens)} if max_tokens else {}),
                    **({"stop_sequences": stop} if stop else {}),
                },
                safety_settings=None,
            )
            # Some SDK versions nest text under resp.text
            return getattr(resp, "text", "").strip()
        except Exception as e:
            return f"[GENERATION_ERROR] {type(e).__name__}: {e}"
