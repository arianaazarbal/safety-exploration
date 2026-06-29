import os
from typing import List
from .base import ChatModel, ChatMessage, ModelConfig

class OpenAIModel(ChatModel):
    def __init__(self, config: ModelConfig):
        self.config = config
        try:
            from openai import OpenAI
        except ImportError as e:
            raise RuntimeError("openai package not installed. pip install openai") from e
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY not set")
        self._client = OpenAI(api_key=api_key)

    def chat(self, messages: List[ChatMessage], **kwargs) -> str:
        # Map to OpenAI format
        oai_msgs = [{"role": m.role, "content": m.content} for m in messages]
        from openai import BadRequestError
        resp = self._client.chat.completions.create(
            model=self.config.model_name,
            messages=oai_msgs,
            temperature=self.config.temperature,
            max_tokens=self.config.max_output_tokens,
        )
        return resp.choices[0].message.content or ""
