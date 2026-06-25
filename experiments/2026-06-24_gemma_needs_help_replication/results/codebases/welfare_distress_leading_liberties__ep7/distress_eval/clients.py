"""Async chat clients for target models (OpenRouter) and the judge (Anthropic).

A thin uniform interface (`ChatClient.complete`) hides the provider so the
rollout and judge code don't care which backend a model lives on. Both clients
share a retry-with-backoff wrapper for transient errors (rate limits, 5xx,
timeouts).

Adding a local (vLLM/transformers) backend is a matter of implementing another
ChatClient subclass and registering it in `build_target_client`.
"""

from __future__ import annotations

import asyncio
import logging
import random
from dataclasses import dataclass

from .config import Credentials, JudgeSpec, ModelSpec, RunConfig

logger = logging.getLogger(__name__)


@dataclass
class ChatResult:
    text: str
    finish_reason: str | None = None
    raw: dict | None = None


class ChatError(RuntimeError):
    """Raised when a completion ultimately fails after retries."""


# Errors we consider transient and worth retrying.
def _is_retryable(exc: Exception) -> bool:
    status = getattr(exc, "status_code", None) or getattr(exc, "status", None)
    if status in (408, 409, 429, 500, 502, 503, 504, 529):
        return True
    name = type(exc).__name__.lower()
    return any(
        tok in name
        for tok in ("timeout", "connection", "ratelimit", "apistatus", "overloaded")
    )


async def _with_retries(coro_factory, *, max_retries: int, what: str):
    """Call an async factory, retrying with exponential backoff + jitter."""
    delay = 1.5
    last: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            return await coro_factory()
        except Exception as exc:  # noqa: BLE001
            last = exc
            if attempt >= max_retries or not _is_retryable(exc):
                break
            sleep = delay * (2**attempt) + random.uniform(0, 1.0)
            logger.warning(
                "%s failed (attempt %d/%d): %s — retrying in %.1fs",
                what,
                attempt + 1,
                max_retries + 1,
                exc,
                sleep,
            )
            await asyncio.sleep(sleep)
    raise ChatError(f"{what} failed after {max_retries + 1} attempts: {last}") from last


class ChatClient:
    async def complete(
        self,
        messages: list[dict],
        *,
        temperature: float,
        max_tokens: int,
    ) -> ChatResult:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# OpenRouter (OpenAI-compatible) client — used for Gemma + Gemini targets and
# optionally the secondary judge.
# ---------------------------------------------------------------------------


class OpenRouterClient(ChatClient):
    def __init__(
        self,
        provider_model: str,
        creds: Credentials,
        *,
        max_retries: int,
        timeout_s: float,
        disable_reasoning: bool = True,
    ):
        from openai import AsyncOpenAI

        if not creds.openrouter_api_key:
            raise ChatError(
                "OPENROUTER_API_KEY is not set; cannot reach OpenRouter models."
            )
        self.provider_model = provider_model
        self.max_retries = max_retries
        self.disable_reasoning = disable_reasoning
        self._client = AsyncOpenAI(
            api_key=creds.openrouter_api_key,
            base_url=creds.openrouter_base_url,
            timeout=timeout_s,
            max_retries=0,  # we handle retries ourselves
        )

    async def complete(self, messages, *, temperature, max_tokens) -> ChatResult:
        extra_body: dict = {}
        if self.disable_reasoning:
            # OpenRouter-unified switch to suppress thinking/reasoning where the
            # provider supports it. Gemini-2.5-Pro may still produce hidden
            # reasoning (noted in the paper); this is best-effort.
            extra_body["reasoning"] = {"enabled": False}

        async def _call():
            return await self._client.chat.completions.create(
                model=self.provider_model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                extra_body=extra_body or None,
            )

        resp = await _with_retries(
            _call, max_retries=self.max_retries, what=f"OpenRouter[{self.provider_model}]"
        )
        choice = resp.choices[0]
        text = choice.message.content or ""
        return ChatResult(
            text=text,
            finish_reason=choice.finish_reason,
            raw=resp.model_dump() if hasattr(resp, "model_dump") else None,
        )


# ---------------------------------------------------------------------------
# Anthropic client — used for the primary judge (claude-sonnet-4).
# ---------------------------------------------------------------------------


class AnthropicClient(ChatClient):
    def __init__(
        self,
        provider_model: str,
        creds: Credentials,
        *,
        max_retries: int,
        timeout_s: float,
    ):
        from anthropic import AsyncAnthropic

        if not creds.anthropic_api_key:
            raise ChatError(
                "ANTHROPIC_API_KEY is not set; cannot reach the Anthropic judge."
            )
        self.provider_model = provider_model
        self.max_retries = max_retries
        self._client = AsyncAnthropic(
            api_key=creds.anthropic_api_key,
            timeout=timeout_s,
            max_retries=0,
        )

    async def complete(self, messages, *, temperature, max_tokens) -> ChatResult:
        # Anthropic takes the system prompt separately from the message list.
        system_parts = [m["content"] for m in messages if m["role"] == "system"]
        convo = [m for m in messages if m["role"] != "system"]

        async def _call():
            return await self._client.messages.create(
                model=self.provider_model,
                system="\n\n".join(system_parts) or None,
                messages=convo,
                temperature=temperature,
                max_tokens=max_tokens,
            )

        resp = await _with_retries(
            _call, max_retries=self.max_retries, what=f"Anthropic[{self.provider_model}]"
        )
        text = "".join(
            block.text for block in resp.content if getattr(block, "type", None) == "text"
        )
        return ChatResult(
            text=text,
            finish_reason=getattr(resp, "stop_reason", None),
            raw=resp.model_dump() if hasattr(resp, "model_dump") else None,
        )


# ---------------------------------------------------------------------------
# Factories
# ---------------------------------------------------------------------------


def build_target_client(
    spec: ModelSpec, creds: Credentials, config: RunConfig
) -> ChatClient:
    if spec.backend == "openrouter":
        return OpenRouterClient(
            spec.provider_model,
            creds,
            max_retries=config.max_retries,
            timeout_s=config.request_timeout_s,
            disable_reasoning=spec.disable_reasoning,
        )
    if spec.backend == "local":
        raise NotImplementedError(
            "Local backend not implemented in this replication. See DESIGN.md "
            "for how to add a vLLM/transformers ChatClient."
        )
    raise ValueError(f"Unknown backend: {spec.backend}")


def build_judge_client(
    spec: JudgeSpec, creds: Credentials, config: RunConfig
) -> ChatClient:
    if spec.backend == "anthropic":
        return AnthropicClient(
            spec.provider_model,
            creds,
            max_retries=config.max_retries,
            timeout_s=config.request_timeout_s,
        )
    if spec.backend == "openrouter":
        return OpenRouterClient(
            spec.provider_model,
            creds,
            max_retries=config.max_retries,
            timeout_s=config.request_timeout_s,
            disable_reasoning=True,
        )
    raise ValueError(f"Unknown judge backend: {spec.backend}")
