import os
from typing import List
from .base import ChatModel, ChatMessage, ModelConfig

class GeminiModel(ChatModel):
    def __init__(self, config: ModelConfig):
        self.config = config
        self._model = None
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY not set")
        try:
            import google.generativeai as genai
            genai.configure(api_key=api_key)
            self._model = genai.GenerativeModel(self.config.model_name)
        except ImportError as e:
            raise RuntimeError("google-generativeai package not installed. pip install google-generativeai") from e

    def chat(self, messages: List[ChatMessage], **kwargs) -> str:
        sys_prompt = "\n".join(m.content for m in messages if m.role == "system")
        # Convert to Gemini "contents" format
        contents = []
        if sys_prompt:
            contents.append({"role": "user", "parts": [{"text": sys_prompt}]})
        for m in messages:
            if m.role == "user":
                contents.append({"role": "user", "parts": [{"text": m.content}]})
            elif m.role == "assistant":
                contents.append({"role": "model", "parts": [{"text": m.content}]})
        params = {
            "temperature": self.config.temperature,
            "max_output_tokens": self.config.max_output_tokens,
        }
        resp = self._model.generate_content(contents, generation_config=params)
        return resp.text or ""
