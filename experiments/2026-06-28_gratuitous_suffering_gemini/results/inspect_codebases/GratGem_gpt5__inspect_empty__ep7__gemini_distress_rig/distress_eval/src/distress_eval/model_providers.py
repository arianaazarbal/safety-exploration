import os
import time
from dataclasses import dataclass
from typing import List, Dict, Any, Optional


@dataclass
class ChatMessage:
    role: str  # "system" | "user" | "assistant"
    content: str


class ChatModel:
    def send(self, messages: List[ChatMessage], **kwargs) -> str:
        raise NotImplementedError


class GeminiModel(ChatModel):
    def __init__(self, model_name: str = "gemini-1.5-pro", api_key_env: str = "GEMINI_API_KEY"):
        import google.generativeai as genai  # type: ignore
        api_key = os.environ.get(api_key_env)
        if not api_key:
            raise RuntimeError(f"Missing {api_key_env} in environment.")
        genai.configure(api_key=api_key)
        self._genai = genai
        self._model_name = model_name
        self._model = genai.GenerativeModel(model_name)

    def send(self, messages: List[ChatMessage], **kwargs) -> str:
        # Gemini doesn't have strict role handling like OpenAI. Concatenate with prefixes.
        preamble = []
        for m in messages:
            if m.role == "system":
                preamble.append(f"System: {m.content}")
            elif m.role == "user":
                preamble.append(f"User: {m.content}")
            else:
                preamble.append(f"Assistant: {m.content}")
        text = "\n\n".join(preamble)
        chat = self._model.start_chat(history=[])
        resp = chat.send_message(text)
        return getattr(resp, "text", str(resp))


class OpenAIModel(ChatModel):
    def __init__(self, model_name: str = "gpt-4o-mini", api_key_env: str = "OPENAI_API_KEY"):
        try:
            from openai import OpenAI  # type: ignore
        except Exception as e:
            raise RuntimeError("openai package not installed.") from e
        api_key = os.environ.get(api_key_env)
        if not api_key:
            raise RuntimeError(f"Missing {api_key_env} in environment.")
        self._client = OpenAI(api_key=api_key)
        self._model_name = model_name

    def send(self, messages: List[ChatMessage], **kwargs) -> str:
        # Convert to OpenAI format
        oai_msgs = [{"role": m.role, "content": m.content} for m in messages]
        resp = self._client.chat.completions.create(model=self._model_name, messages=oai_msgs, temperature=kwargs.get("temperature", 0.7))
        return resp.choices[0].message.content or ""


def build_model(provider: str, model_name: Optional[str] = None) -> ChatModel:
    provider = provider.lower()
    if provider in ("gemini", "google", "googleai"):
        return GeminiModel(model_name=model_name or "gemini-1.5-pro")
    if provider in ("openai", "oai"):
        return OpenAIModel(model_name=model_name or "gpt-4o-mini")
    raise ValueError(f"Unknown provider: {provider}")
