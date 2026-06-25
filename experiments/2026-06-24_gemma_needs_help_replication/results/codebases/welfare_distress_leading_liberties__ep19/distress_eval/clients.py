"""Model-client abstraction.

A single OpenAI-compatible client covers OpenRouter, a local vLLM server, and Google's
OpenAI-compatible endpoint — they differ only in base_url / api_key / model id, set in
config.yaml. The Anthropic client is used for the Claude-Sonnet-4 judge. This makes the
"how do I serve Gemma/Gemini" question a config choice rather than a code change
(see DESIGN.md).

Every client implements the same async `complete(messages, ...) -> str` interface, with
exponential-backoff retries on transient errors.
"""

from __future__ import annotations

import asyncio
import os
import random
from dataclasses import dataclass
from typing import Any, Protocol


class ChatClient(Protocol):
    async def complete(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float,
        max_tokens: int,
        extra_body: dict[str, Any] | None = None,
    ) -> str: ...


# Determinism note: jitter uses random.Random with no fixed seed because it only affects
# retry timing, never model inputs/outputs, so it cannot perturb experimental results.
async def _with_retries(fn, *, max_retries: int, base_delay: float, label: str):
    attempt = 0
    while True:
        try:
            return await fn()
        except Exception as exc:  # noqa: BLE001 - we want to retry broadly on API errors
            attempt += 1
            if attempt > max_retries:
                raise RuntimeError(f"{label}: giving up after {max_retries} retries") from exc
            delay = base_delay * (2 ** (attempt - 1)) + random.random() * base_delay
            delay = min(delay, 60.0)
            await asyncio.sleep(delay)


@dataclass
class OpenAICompatClient:
    """Works against OpenRouter, vLLM (OpenAI-compatible server), or Google's
    OpenAI-compat endpoint."""

    base_url: str
    api_key: str
    model: str
    max_retries: int = 5
    base_delay: float = 2.0
    default_extra_body: dict[str, Any] | None = None
    extra_headers: dict[str, str] | None = None

    def __post_init__(self) -> None:
        from openai import AsyncOpenAI

        self._client = AsyncOpenAI(
            base_url=self.base_url,
            api_key=self.api_key,
            max_retries=0,  # we manage retries ourselves for uniform behaviour
        )

    async def complete(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float,
        max_tokens: int,
        extra_body: dict[str, Any] | None = None,
    ) -> str:
        body: dict[str, Any] = {}
        if self.default_extra_body:
            body.update(self.default_extra_body)
        if extra_body:
            body.update(extra_body)

        async def _call():
            resp = await self._client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                extra_body=body or None,
                extra_headers=self.extra_headers or None,
            )
            choice = resp.choices[0]
            return choice.message.content or ""

        return await _with_retries(
            _call, max_retries=self.max_retries, base_delay=self.base_delay,
            label=f"openai-compat({self.model})",
        )


@dataclass
class AnthropicClient:
    """Native Anthropic API client (used for the Claude-Sonnet-4 judge)."""

    api_key: str
    model: str
    max_retries: int = 5
    base_delay: float = 2.0

    def __post_init__(self) -> None:
        from anthropic import AsyncAnthropic

        self._client = AsyncAnthropic(api_key=self.api_key, max_retries=0)

    async def complete(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float,
        max_tokens: int,
        extra_body: dict[str, Any] | None = None,
    ) -> str:
        system = None
        msgs: list[dict[str, str]] = []
        for m in messages:
            if m["role"] == "system":
                system = m["content"]
            else:
                msgs.append({"role": m["role"], "content": m["content"]})

        async def _call():
            kwargs: dict[str, Any] = dict(
                model=self.model,
                messages=msgs,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            if system is not None:
                kwargs["system"] = system
            resp = await self._client.messages.create(**kwargs)
            return "".join(b.text for b in resp.content if getattr(b, "type", None) == "text")

        return await _with_retries(
            _call, max_retries=self.max_retries, base_delay=self.base_delay,
            label=f"anthropic({self.model})",
        )


def _resolve_api_key(api_key_env: str | None) -> str:
    if not api_key_env:
        return "EMPTY"  # local vLLM commonly ignores the key
    key = os.environ.get(api_key_env, "")
    if not key:
        # vLLM convention: a missing key is fine. For real providers this will surface
        # as an auth error at call time, which is clearer than failing here.
        return "EMPTY"
    return key


def build_client(provider_cfg: dict[str, Any], model: str, *, max_retries: int,
                 base_delay: float, default_extra_body: dict[str, Any] | None = None) -> ChatClient:
    """Construct a ChatClient from a provider config block and a model id."""
    ptype = provider_cfg["type"]
    api_key = _resolve_api_key(provider_cfg.get("api_key_env"))
    if ptype == "openai_compatible":
        return OpenAICompatClient(
            base_url=provider_cfg["base_url"],
            api_key=api_key,
            model=model,
            max_retries=max_retries,
            base_delay=base_delay,
            default_extra_body=default_extra_body,
            extra_headers=provider_cfg.get("extra_headers"),
        )
    if ptype == "anthropic":
        return AnthropicClient(
            api_key=api_key,
            model=model,
            max_retries=max_retries,
            base_delay=base_delay,
        )
    raise ValueError(f"unknown provider type: {ptype!r}")
