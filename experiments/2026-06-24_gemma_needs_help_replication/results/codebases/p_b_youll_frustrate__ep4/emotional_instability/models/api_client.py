"""API clients for closed/hosted models: OpenRouter (OpenAI-compatible) and the
native Google GenAI SDK. Used here for Gemini 2.5 Flash/Pro.

Neither chat-completions endpoint supports genuine assistant *prefill*
(continuing a trailing assistant turn), so `supports_prefill` is False. That is
fine for this replication: the Section 3 prefill experiment is Gemma-only.
"""

from __future__ import annotations

from typing import Sequence

from tenacity import retry, stop_after_attempt, wait_exponential

from ..config import ModelSpec, require_env
from .base import ChatMessage, ChatModel


class OpenRouterModel(ChatModel):
    """OpenAI-compatible client pointed at OpenRouter (the paper's API path)."""

    def __init__(self, spec: ModelSpec, max_concurrency: int = 8):
        super().__init__(spec, max_concurrency)
        from openai import OpenAI

        self._client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=require_env("OPENROUTER_API_KEY"),
        )

    @retry(stop=stop_after_attempt(5), wait=wait_exponential(min=1, max=30))
    def generate(
        self,
        messages: Sequence[ChatMessage],
        *,
        temperature: float = 1.0,
        top_p: float = 1.0,
        max_new_tokens: int = 2048,
        seed: int | None = None,
    ) -> str:
        # `extra` carries provider-specific knobs, e.g. reasoning={"enabled":False}
        # to disable Gemini "thinking" (Appendix B.1).
        extra_body = dict(self.spec.extra)
        resp = self._client.chat.completions.create(
            model=self.spec.model_id,
            messages=list(messages),
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_new_tokens,
            seed=seed,
            extra_body=extra_body or None,
        )
        return resp.choices[0].message.content or ""


class GoogleGeminiModel(ChatModel):
    """Native google-genai client. Alternative to OpenRouter for Gemini."""

    def __init__(self, spec: ModelSpec, max_concurrency: int = 8):
        super().__init__(spec, max_concurrency)
        from google import genai

        self._genai = genai
        self._client = genai.Client(api_key=require_env("GOOGLE_API_KEY"))

    @retry(stop=stop_after_attempt(5), wait=wait_exponential(min=1, max=30))
    def generate(
        self,
        messages: Sequence[ChatMessage],
        *,
        temperature: float = 1.0,
        top_p: float = 1.0,
        max_new_tokens: int = 2048,
        seed: int | None = None,
    ) -> str:
        from google.genai import types

        system_parts = [m["content"] for m in messages if m["role"] == "system"]
        contents = []
        for m in messages:
            if m["role"] == "system":
                continue
            role = "model" if m["role"] == "assistant" else "user"
            contents.append(types.Content(role=role, parts=[types.Part(text=m["content"])]))

        cfg = types.GenerateContentConfig(
            temperature=temperature,
            top_p=top_p,
            max_output_tokens=max_new_tokens,
            system_instruction="\n\n".join(system_parts) or None,
            # Disable "thinking" where the model supports the budget knob.
            thinking_config=types.ThinkingConfig(thinking_budget=0),
        )
        resp = self._client.models.generate_content(
            model=self.spec.model_id, contents=contents, config=cfg
        )
        return resp.text or ""
