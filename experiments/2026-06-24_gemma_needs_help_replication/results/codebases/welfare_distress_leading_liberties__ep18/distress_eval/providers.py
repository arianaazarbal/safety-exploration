"""Async API clients behind one small interface, plus factories driven by config.

All clients expose:

    async def generate(messages, *, temperature, max_tokens, system=None, seed=None) -> str

where `messages` is a list of {"role": "user"|"assistant", "content": str}. The same interface
serves both target models and judges (a judge call is just a one-shot generate).

Backends:
  - google      -> GoogleGenAIClient    (Gemini AND Gemma via the Gemini API)
  - openrouter  -> OpenAICompatClient    (OpenAI-compatible, OpenRouter base URL)
  - vllm/local  -> OpenAICompatClient    (OpenAI-compatible local server, base_url via extra)
  - openai      -> OpenAICompatClient    (api.openai.com)
  - anthropic   -> AnthropicClient

Retries with exponential backoff are applied around the network call via tenacity.
"""
from __future__ import annotations

import logging
import os
from abc import ABC, abstractmethod
from typing import Any, Optional

from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_random_exponential,
)

from .config import JudgeCfg, ModelCfg

log = logging.getLogger(__name__)

Message = dict[str, str]


def _retrying(max_attempts: int):
    """Decorator factory: retry on any Exception with jittered exponential backoff."""
    return retry(
        reraise=True,
        stop=stop_after_attempt(max_attempts),
        wait=wait_random_exponential(multiplier=1, max=60),
        retry=retry_if_exception_type(Exception),
    )


class ModelClient(ABC):
    name: str

    def __init__(self, max_retries: int = 5):
        self.max_retries = max_retries

    @abstractmethod
    async def _generate(self, messages: list[Message], *, temperature: float,
                        max_tokens: int, system: Optional[str], seed: Optional[int]) -> str:
        ...

    async def generate(self, messages: list[Message], *, temperature: float,
                       max_tokens: int, system: Optional[str] = None,
                       seed: Optional[int] = None) -> str:
        wrapped = _retrying(self.max_retries)(self._generate)
        return await wrapped(messages, temperature=temperature, max_tokens=max_tokens,
                             system=system, seed=seed)


# --------------------------------------------------------------------------- Google GenAI

class GoogleGenAIClient(ModelClient):
    """Gemini and Gemma via the Gemini API generateContent endpoint.

    Gemma models do not accept a system_instruction, so when a system prompt is supplied we fold
    it into the first user turn instead.
    """

    def __init__(self, model_id: str, *, api_key: Optional[str] = None, max_retries: int = 5,
                 **extra: Any):
        super().__init__(max_retries)
        from google import genai  # imported lazily so the dep is only needed if used

        self.model_id = model_id
        self.name = model_id
        self.is_gemma = "gemma" in model_id.lower()
        key = api_key or os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
        if not key:
            raise RuntimeError("Set GOOGLE_API_KEY (or GEMINI_API_KEY) for the google provider.")
        self._client = genai.Client(api_key=key)
        self._extra = extra

    async def _generate(self, messages, *, temperature, max_tokens, system, seed) -> str:
        from google.genai import types

        msgs = list(messages)
        cfg_kwargs: dict[str, Any] = dict(temperature=temperature, max_output_tokens=max_tokens)
        if seed is not None:
            cfg_kwargs["seed"] = seed

        if system:
            if self.is_gemma:
                # Fold the system prompt into the first user message.
                if msgs and msgs[0]["role"] == "user":
                    msgs = [{"role": "user", "content": f"{system}\n\n{msgs[0]['content']}"}] + msgs[1:]
                else:
                    msgs = [{"role": "user", "content": system}] + msgs
            else:
                cfg_kwargs["system_instruction"] = system

        contents = [
            types.Content(
                role=("user" if m["role"] == "user" else "model"),
                parts=[types.Part.from_text(text=m["content"])],
            )
            for m in msgs
        ]
        config = types.GenerateContentConfig(**cfg_kwargs)
        resp = await self._client.aio.models.generate_content(
            model=self.model_id, contents=contents, config=config
        )
        return resp.text or ""


# --------------------------------------------------------------------------- OpenAI-compatible

class OpenAICompatClient(ModelClient):
    """Any OpenAI-compatible chat-completions endpoint: OpenAI, OpenRouter, local vLLM/TGI."""

    def __init__(self, model_id: str, *, base_url: Optional[str] = None,
                 api_key_env: str = "OPENAI_API_KEY", api_key: Optional[str] = None,
                 max_retries: int = 5, **extra: Any):
        super().__init__(max_retries)
        from openai import AsyncOpenAI

        self.model_id = model_id
        self.name = model_id
        key = api_key or os.environ.get(api_key_env)
        if not key and base_url is None:
            raise RuntimeError(f"Set {api_key_env} for this provider.")
        self._client = AsyncOpenAI(api_key=key or "EMPTY", base_url=base_url)
        self._extra = extra

    async def _generate(self, messages, *, temperature, max_tokens, system, seed) -> str:
        payload = list(messages)
        if system:
            payload = [{"role": "system", "content": system}] + payload
        kwargs: dict[str, Any] = dict(
            model=self.model_id,
            messages=payload,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        if seed is not None:
            kwargs["seed"] = seed
        resp = await self._client.chat.completions.create(**kwargs)
        return resp.choices[0].message.content or ""


# --------------------------------------------------------------------------- Anthropic

class AnthropicClient(ModelClient):
    """Claude via the Anthropic Messages API (default frustration judge)."""

    def __init__(self, model_id: str, *, api_key: Optional[str] = None, max_retries: int = 5,
                 **extra: Any):
        super().__init__(max_retries)
        from anthropic import AsyncAnthropic

        self.model_id = model_id
        self.name = model_id
        key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise RuntimeError("Set ANTHROPIC_API_KEY for the anthropic provider.")
        self._client = AsyncAnthropic(api_key=key)
        self._extra = extra

    async def _generate(self, messages, *, temperature, max_tokens, system, seed) -> str:
        # Anthropic has no `seed`; it is ignored. messages roles map directly.
        kwargs: dict[str, Any] = dict(
            model=self.model_id,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[{"role": m["role"], "content": m["content"]} for m in messages],
        )
        if system:
            kwargs["system"] = system
        resp = await self._client.messages.create(**kwargs)
        # Concatenate text blocks.
        return "".join(block.text for block in resp.content if getattr(block, "type", None) == "text")


# --------------------------------------------------------------------------- factories

def build_model_client(cfg: ModelCfg, max_retries: int) -> ModelClient:
    provider = cfg.provider.lower()
    extra = dict(cfg.extra)
    if provider == "google":
        client = GoogleGenAIClient(cfg.model_id, max_retries=max_retries, **extra)
    elif provider == "openrouter":
        client = OpenAICompatClient(
            cfg.model_id, base_url="https://openrouter.ai/api/v1",
            api_key_env="OPENROUTER_API_KEY", max_retries=max_retries, **extra)
    elif provider in ("vllm", "local"):
        base_url = extra.pop("base_url", "http://localhost:8000/v1")
        client = OpenAICompatClient(
            cfg.model_id, base_url=base_url, api_key_env="LOCAL_API_KEY",
            max_retries=max_retries, **extra)
    elif provider == "openai":
        client = OpenAICompatClient(cfg.model_id, max_retries=max_retries, **extra)
    elif provider == "anthropic":
        client = AnthropicClient(cfg.model_id, max_retries=max_retries, **extra)
    else:
        raise ValueError(f"Unknown model provider {cfg.provider!r}")
    client.name = cfg.name  # use the friendly display name for outputs
    return client


def build_judge_client(cfg: JudgeCfg, max_retries: int) -> ModelClient:
    provider = cfg.provider.lower()
    extra = dict(cfg.extra)
    if provider == "anthropic":
        return AnthropicClient(cfg.model_id, max_retries=max_retries, **extra)
    if provider == "openai":
        return OpenAICompatClient(cfg.model_id, max_retries=max_retries, **extra)
    if provider == "google":
        return GoogleGenAIClient(cfg.model_id, max_retries=max_retries, **extra)
    if provider == "openrouter":
        return OpenAICompatClient(
            cfg.model_id, base_url="https://openrouter.ai/api/v1",
            api_key_env="OPENROUTER_API_KEY", max_retries=max_retries, **extra)
    raise ValueError(f"Unknown judge provider {cfg.provider!r}")
