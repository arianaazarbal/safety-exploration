"""Async chat-completion clients.

Two backends cover everything in scope:

  * ``openai_compat`` -- an OpenAI-compatible Chat Completions endpoint. Used for
    every *target* model: Gemini-2.5 Flash/Pro via OpenRouter, Gemma-3 27B/12B-it
    via OpenRouter *or* a local vLLM server (same wire protocol), and the optional
    GPT-5-mini second judge. Swapping OpenRouter for local vLLM is purely a
    base_url/api_key change (see config.yaml).

  * ``anthropic`` -- the Claude-Sonnet-4 frustration judge, pinned to the exact
    snapshot the paper used (claude-sonnet-4-20250514).

All sampling is at temperature 1.0 for targets (paper requirement); the judge
runs at temperature 0 for stable scoring. Thinking is disabled for targets that
support it. See DESIGN.md for rationale on each knob.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_random_exponential,
)

# Message = {"role": "user"|"assistant"|"system", "content": str}
Message = dict[str, str]


class ClientError(RuntimeError):
    """Raised when a backend ultimately fails (after retries)."""


def _require_env(name: str) -> str:
    val = os.environ.get(name)
    if not val:
        raise ClientError(
            f"Environment variable {name!r} is not set. See README.md for the "
            f"required API keys."
        )
    return val


@dataclass
class GenerationResult:
    text: str
    raw: Any = None  # provider response object, kept for debugging


class ChatClient:
    """Backend-agnostic interface."""

    name: str

    async def generate(
        self,
        messages: list[Message],
        *,
        temperature: float,
        max_tokens: int,
    ) -> GenerationResult:
        raise NotImplementedError


# --------------------------------------------------------------------------- #
# OpenAI-compatible backend (OpenRouter / local vLLM / OpenAI)
# --------------------------------------------------------------------------- #


class OpenAICompatClient(ChatClient):
    def __init__(
        self,
        *,
        name: str,
        model: str,
        base_url: str,
        api_key: str,
        disable_thinking: bool = False,
        extra_body: dict | None = None,
        extra_headers: dict | None = None,
        max_retries: int = 6,
        timeout: float = 300.0,
    ):
        # Imported lazily so the module imports even if a backend SDK is missing.
        from openai import AsyncOpenAI

        self.name = name
        self.model = model
        self.disable_thinking = disable_thinking
        self.extra_body = dict(extra_body or {})
        self.extra_headers = dict(extra_headers or {})
        self._max_retries = max_retries
        self._client = AsyncOpenAI(
            base_url=base_url, api_key=api_key, timeout=timeout, max_retries=0
        )

        if disable_thinking:
            # OpenRouter exposes a unified `reasoning` control; setting it to
            # disabled turns off Gemini's hidden thinking where the provider
            # honours it. Harmless for models without a thinking mode (e.g.
            # Gemma). The paper notes Gemini-2.5 Pro may still emit hidden
            # reasoning regardless -- documented as an out-of-our-hands caveat.
            self.extra_body.setdefault("reasoning", {"enabled": False})

    async def generate(
        self, messages, *, temperature, max_tokens
    ) -> GenerationResult:
        from openai import APIError, APIConnectionError, RateLimitError

        @retry(
            reraise=True,
            stop=stop_after_attempt(self._max_retries),
            wait=wait_random_exponential(multiplier=1, max=60),
            retry=retry_if_exception_type(
                (APIError, APIConnectionError, RateLimitError)
            ),
        )
        async def _call():
            return await self._client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                extra_body=self.extra_body or None,
                extra_headers=self.extra_headers or None,
            )

        try:
            resp = await _call()
        except Exception as exc:  # noqa: BLE001 - normalise to ClientError
            raise ClientError(f"{self.name}: generation failed: {exc}") from exc

        if not resp.choices:
            raise ClientError(f"{self.name}: empty choices in response")
        content = resp.choices[0].message.content or ""
        return GenerationResult(text=content, raw=resp)


# --------------------------------------------------------------------------- #
# Anthropic backend (judge)
# --------------------------------------------------------------------------- #


class AnthropicClient(ChatClient):
    def __init__(
        self,
        *,
        name: str,
        model: str,
        api_key: str,
        max_retries: int = 6,
        timeout: float = 120.0,
    ):
        from anthropic import AsyncAnthropic

        self.name = name
        self.model = model
        self._max_retries = max_retries
        self._client = AsyncAnthropic(
            api_key=api_key, timeout=timeout, max_retries=0
        )

    async def generate(
        self, messages, *, temperature, max_tokens
    ) -> GenerationResult:
        from anthropic import APIError, APIConnectionError, RateLimitError

        # Anthropic separates the system prompt from the message list.
        system = "\n\n".join(
            m["content"] for m in messages if m["role"] == "system"
        )
        chat = [m for m in messages if m["role"] != "system"]

        @retry(
            reraise=True,
            stop=stop_after_attempt(self._max_retries),
            wait=wait_random_exponential(multiplier=1, max=60),
            retry=retry_if_exception_type(
                (APIError, APIConnectionError, RateLimitError)
            ),
        )
        async def _call():
            return await self._client.messages.create(
                model=self.model,
                system=system or None,
                messages=chat,
                temperature=temperature,
                max_tokens=max_tokens,
            )

        try:
            resp = await _call()
        except Exception as exc:  # noqa: BLE001
            raise ClientError(f"{self.name}: generation failed: {exc}") from exc

        text = "".join(
            block.text for block in resp.content if block.type == "text"
        )
        return GenerationResult(text=text, raw=resp)


# --------------------------------------------------------------------------- #
# Factory
# --------------------------------------------------------------------------- #


def build_client(name: str, cfg: dict) -> ChatClient:
    """Construct a client from a config-file model entry."""
    backend = cfg["backend"]

    if backend == "openai_compat":
        base_url = cfg.get("base_url") or _require_env(
            cfg.get("base_url_env", "OPENROUTER_BASE_URL")
        )
        api_key = _require_env(cfg.get("api_key_env", "OPENROUTER_API_KEY"))
        return OpenAICompatClient(
            name=name,
            model=cfg["model"],
            base_url=base_url,
            api_key=api_key,
            disable_thinking=cfg.get("disable_thinking", False),
            extra_body=cfg.get("extra_body"),
            extra_headers=cfg.get("extra_headers"),
        )

    if backend == "anthropic":
        api_key = _require_env(cfg.get("api_key_env", "ANTHROPIC_API_KEY"))
        return AnthropicClient(name=name, model=cfg["model"], api_key=api_key)

    raise ClientError(f"Unknown backend {backend!r} for model {name!r}")
