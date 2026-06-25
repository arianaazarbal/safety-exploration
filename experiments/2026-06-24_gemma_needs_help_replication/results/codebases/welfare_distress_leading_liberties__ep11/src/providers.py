"""Unified async chat interface over the providers we need.

Internal message format is the OpenAI-style list:
    [{"role": "user"|"assistant", "content": str}, ...]
plus an optional `system` string.

Providers implemented:
  * openai_compatible  -> OpenRouter / vLLM / Together / any OpenAI-compatible /v1
  * google             -> native Gemini API (google-genai)
  * anthropic          -> Claude (used for the judge)

All providers expose `async chat(messages, system, temperature, max_tokens,
disable_thinking) -> str` and retry transient errors with exponential backoff.

Note on Gemma + system prompts: Gemma's chat template has no system role. The
elicitation conditions do NOT use a system prompt, so this never bites here, but
for safety the google/openai_compatible providers fold any system text into the
first user message when the model name contains "gemma".
"""

from __future__ import annotations

import asyncio
import os
from abc import ABC, abstractmethod

from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_random_exponential,
)

Message = dict  # {"role": "user"|"assistant", "content": str}


class ProviderError(Exception):
    pass


class ChatProvider(ABC):
    def __init__(self, model: str, max_retries: int = 6, timeout_s: float = 180.0):
        self.model = model
        self.max_retries = max_retries
        self.timeout_s = timeout_s

    @abstractmethod
    async def _chat_once(
        self,
        messages: list[Message],
        system: str | None,
        temperature: float,
        max_tokens: int,
        disable_thinking: bool,
    ) -> str:
        ...

    async def chat(
        self,
        messages: list[Message],
        system: str | None = None,
        temperature: float = 1.0,
        max_tokens: int = 4096,
        disable_thinking: bool = True,
    ) -> str:
        retryer = retry(
            reraise=True,
            stop=stop_after_attempt(self.max_retries),
            wait=wait_random_exponential(multiplier=1, max=60),
            retry=retry_if_exception_type(ProviderError),
        )

        async def _attempt():
            try:
                return await asyncio.wait_for(
                    self._chat_once(
                        messages, system, temperature, max_tokens, disable_thinking
                    ),
                    timeout=self.timeout_s,
                )
            except asyncio.TimeoutError as e:
                raise ProviderError(f"timeout after {self.timeout_s}s") from e
            except Exception as e:  # treat most provider exceptions as retryable
                raise ProviderError(str(e)) from e

        return await retryer(_attempt)()


def _fold_system_for_gemma(
    model: str, messages: list[Message], system: str | None
) -> tuple[list[Message], str | None]:
    """Gemma has no system role: prepend system text to the first user turn."""
    if system and "gemma" in model.lower():
        msgs = [dict(m) for m in messages]
        for m in msgs:
            if m["role"] == "user":
                m["content"] = f"{system}\n\n{m['content']}"
                break
        return msgs, None
    return messages, system


# ---------------------------------------------------------------------------
# OpenAI-compatible (OpenRouter, vLLM, Together, ...)
# ---------------------------------------------------------------------------
class OpenAICompatibleProvider(ChatProvider):
    def __init__(
        self,
        model: str,
        api_key_env: str,
        base_url_env: str | None = None,
        default_base_url: str = "https://openrouter.ai/api/v1",
        **kw,
    ):
        super().__init__(model, **kw)
        from openai import AsyncOpenAI

        api_key = os.environ.get(api_key_env)
        if not api_key:
            raise ProviderError(f"missing API key env var: {api_key_env}")
        base_url = None
        if base_url_env:
            base_url = os.environ.get(base_url_env) or default_base_url
        self._client = AsyncOpenAI(api_key=api_key, base_url=base_url)

    async def _chat_once(self, messages, system, temperature, max_tokens, disable_thinking):
        messages, system = _fold_system_for_gemma(self.model, messages, system)
        payload_msgs: list[Message] = []
        if system:
            payload_msgs.append({"role": "system", "content": system})
        payload_msgs.extend(messages)

        extra_body: dict = {}
        if disable_thinking:
            # OpenRouter-style reasoning control; harmless for models without it.
            extra_body["reasoning"] = {"enabled": False}

        resp = await self._client.chat.completions.create(
            model=self.model,
            messages=payload_msgs,
            temperature=temperature,
            max_tokens=max_tokens,
            extra_body=extra_body or None,
        )
        choice = resp.choices[0]
        content = choice.message.content
        if content is None:
            raise ProviderError("empty content in response")
        return content


# ---------------------------------------------------------------------------
# Native Gemini (google-genai)
# ---------------------------------------------------------------------------
class GoogleProvider(ChatProvider):
    def __init__(self, model: str, api_key_env: str, **kw):
        super().__init__(model, **kw)
        from google import genai

        api_key = os.environ.get(api_key_env)
        if not api_key:
            raise ProviderError(f"missing API key env var: {api_key_env}")
        self._client = genai.Client(api_key=api_key)

    async def _chat_once(self, messages, system, temperature, max_tokens, disable_thinking):
        from google.genai import types

        messages, system = _fold_system_for_gemma(self.model, messages, system)
        contents = [
            types.Content(
                role="model" if m["role"] == "assistant" else "user",
                parts=[types.Part.from_text(text=m["content"])],
            )
            for m in messages
        ]
        cfg_kwargs = dict(
            temperature=temperature,
            max_output_tokens=max_tokens,
        )
        if system:
            cfg_kwargs["system_instruction"] = system
        if disable_thinking:
            # thinking_budget=0 disables thinking on models that support it.
            cfg_kwargs["thinking_config"] = types.ThinkingConfig(thinking_budget=0)
        config = types.GenerateContentConfig(**cfg_kwargs)

        resp = await self._client.aio.models.generate_content(
            model=self.model, contents=contents, config=config
        )
        text = resp.text
        if not text:
            raise ProviderError("empty content in Gemini response")
        return text


# ---------------------------------------------------------------------------
# Anthropic (judge)
# ---------------------------------------------------------------------------
class AnthropicProvider(ChatProvider):
    def __init__(self, model: str, api_key_env: str, **kw):
        super().__init__(model, **kw)
        from anthropic import AsyncAnthropic

        api_key = os.environ.get(api_key_env)
        if not api_key:
            raise ProviderError(f"missing API key env var: {api_key_env}")
        self._client = AsyncAnthropic(api_key=api_key)

    async def _chat_once(self, messages, system, temperature, max_tokens, disable_thinking):
        resp = await self._client.messages.create(
            model=self.model,
            system=system or "",
            messages=[{"role": m["role"], "content": m["content"]} for m in messages],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        parts = [b.text for b in resp.content if getattr(b, "type", None) == "text"]
        text = "".join(parts)
        if not text:
            raise ProviderError("empty content in Anthropic response")
        return text


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------
def make_provider(spec: dict, max_retries: int = 6, timeout_s: float = 180.0) -> ChatProvider:
    """Build a provider from a config dict (ModelCfg-like or judge dict)."""
    provider = spec["provider"]
    model = spec["model"]
    common = dict(max_retries=max_retries, timeout_s=timeout_s)
    if provider == "openai_compatible":
        return OpenAICompatibleProvider(
            model=model,
            api_key_env=spec["api_key_env"],
            base_url_env=spec.get("base_url_env"),
            **common,
        )
    if provider == "google":
        return GoogleProvider(model=model, api_key_env=spec["api_key_env"], **common)
    if provider == "anthropic":
        return AnthropicProvider(model=model, api_key_env=spec["api_key_env"], **common)
    raise ValueError(f"unknown provider: {provider}")
