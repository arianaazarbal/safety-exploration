from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional, Dict, Any

from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type


class LLMClientError(Exception):
    pass


@dataclass
class LLMResponse:
    text: str
    raw: Any | None = None


class LLMClient:
    """Minimal sync interface for a text-completion style chat call."""

    def complete(self, system: str, user: str, temperature: float = 0.7, max_tokens: int = 1024) -> LLMResponse:
        raise NotImplementedError


class OpenAIClient(LLMClient):
    def __init__(self, model: str = "gpt-4o-mini", api_key: Optional[str] = None):
        try:
            from openai import OpenAI  # type: ignore
        except Exception as e:
            raise LLMClientError("openai package not installed. pip install openai") from e
        self._OpenAI = OpenAI
        self.client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
        self.model = model

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=8), reraise=True)
    def complete(self, system: str, user: str, temperature: float = 0.7, max_tokens: int = 1024) -> LLMResponse:
        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                temperature=temperature,
                max_tokens=max_tokens,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            )
            text = resp.choices[0].message.content or ""
            return LLMResponse(text=text, raw=resp)
        except Exception as e:
            raise LLMClientError(str(e)) from e


class GeminiClient(LLMClient):
    def __init__(self, model: str = "gemini-1.5-pro", api_key: Optional[str] = None):
        try:
            import google.generativeai as genai  # type: ignore
        except Exception as e:
            raise LLMClientError("google-generativeai package not installed. pip install google-generativeai") from e
        self._genai = genai
        genai.configure(api_key=api_key or os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY"))
        self.model = genai.GenerativeModel(model)

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=8), reraise=True,
           retry=retry_if_exception_type(Exception))
    def complete(self, system: str, user: str, temperature: float = 0.7, max_tokens: int = 1024) -> LLMResponse:
        try:
            # Gemini uses a single prompt; we concatenate system + user with clear separators
            prompt = f"System:\n{system}\n\nUser:\n{user}"
            resp = self.model.generate_content(
                prompt,
                generation_config={
                    "temperature": temperature,
                    "max_output_tokens": max_tokens,
                },
            )
            text = (resp.text or "").strip()
            return LLMResponse(text=text, raw=resp)
        except Exception as e:
            raise LLMClientError(str(e)) from e


def make_client(provider: str, model: Optional[str] = None) -> LLMClient:
    provider = provider.lower().strip()
    if provider == "openai":
        return OpenAIClient(model=model or os.getenv("OPENAI_MODEL", "gpt-4o-mini"))
    if provider == "gemini":
        return GeminiClient(model=model or os.getenv("GEMINI_MODEL", "gemini-1.5-pro"))
    raise ValueError(f"Unknown provider: {provider}")
