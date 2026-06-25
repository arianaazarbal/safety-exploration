"""OpenAI-compatible async backend.

Covers both OpenRouter (hosted Gemini / judges) and a local vLLM server (Gemma,
including base models and prefill). The wire format is the OpenAI `/v1/chat/completions`
and `/v1/completions` API in both cases; only base_url / headers differ.

Robustness features for unattended multi-week operation:
  * per-backend concurrency semaphore
  * exponential backoff with jitter on 429/5xx/timeouts/connection errors (tenacity)
  * honours Retry-After when present
  * distinguishes retryable transport errors from terminal 4xx (bad request)
"""
from __future__ import annotations

import asyncio
import random
from typing import Any

import httpx
from tenacity import (
    AsyncRetrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential_jitter,
)

from ..config import BackendConfig
from ..logging_utils import get_logger
from .base import BackendError, ChatBackend, GenResult, Message

log = get_logger(__name__)


class RetryableHTTP(Exception):
    """Wraps a transient failure so tenacity retries it."""

    def __init__(self, msg: str, retry_after: float | None = None):
        super().__init__(msg)
        self.retry_after = retry_after


_RETRYABLE_STATUS = {408, 409, 425, 429, 500, 502, 503, 504, 520, 522, 524}


class OpenAICompatBackend(ChatBackend):
    def __init__(self, name: str, cfg: BackendConfig, max_attempts: int = 8):
        self.name = name
        self.cfg = cfg
        self.max_attempts = max_attempts
        self._sem = asyncio.Semaphore(cfg.max_concurrency)
        headers = {
            "Authorization": f"Bearer {cfg.api_key()}",
            "Content-Type": "application/json",
            **cfg.default_extra_headers,
        }
        self._client = httpx.AsyncClient(
            base_url=cfg.base_url,
            headers=headers,
            timeout=httpx.Timeout(cfg.timeout_s),
            limits=httpx.Limits(
                max_connections=cfg.max_concurrency * 2,
                max_keepalive_connections=cfg.max_concurrency,
            ),
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    # ------------------------------------------------------------------ public
    async def chat(
        self,
        model_id: str,
        messages: list[Message],
        *,
        temperature: float,
        max_tokens: int,
        prefill: str | None = None,
        extra_body: dict[str, Any] | None = None,
    ) -> GenResult:
        payload: dict[str, Any] = {
            "model": model_id,
            "messages": [m.as_dict() for m in messages],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if prefill is not None:
            # Continue from an assistant prefix. vLLM honours these flags; the prefix is
            # appended as a partial assistant message that the model continues.
            payload["messages"].append({"role": "assistant", "content": prefill})
            payload["add_generation_prompt"] = False
            payload["continue_final_message"] = True
        if extra_body:
            payload.update(extra_body)
        data = await self._post("/chat/completions", payload)
        choice = data["choices"][0]
        msg = choice.get("message", {})
        return self._mk_result(choice, msg.get("content") or "", data)

    async def complete(
        self,
        model_id: str,
        prompt: str,
        *,
        temperature: float,
        max_tokens: int,
        extra_body: dict[str, Any] | None = None,
    ) -> GenResult:
        payload: dict[str, Any] = {
            "model": model_id,
            "prompt": prompt,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if extra_body:
            payload.update(extra_body)
        data = await self._post("/completions", payload)
        choice = data["choices"][0]
        return self._mk_result(choice, choice.get("text") or "", data)

    # ----------------------------------------------------------------- helpers
    @staticmethod
    def _mk_result(choice: dict, text: str, data: dict) -> GenResult:
        usage = data.get("usage") or {}
        return GenResult(
            text=text,
            finish_reason=choice.get("finish_reason"),
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=usage.get("completion_tokens", 0),
            raw=data,
        )

    async def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        retryer = AsyncRetrying(
            stop=stop_after_attempt(self.max_attempts),
            wait=wait_exponential_jitter(initial=2, max=120),
            retry=retry_if_exception_type((RetryableHTTP,)),
            reraise=True,
        )
        try:
            async for attempt in retryer:
                with attempt:
                    return await self._post_once(path, payload)
        except RetryableHTTP as e:  # exhausted retries on a transient error
            raise BackendError(f"{self.name}: retries exhausted: {e}") from e
        raise BackendError(f"{self.name}: unreachable")  # pragma: no cover

    async def _post_once(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        async with self._sem:
            try:
                resp = await self._client.post(path, json=payload)
            except (httpx.TimeoutException, httpx.TransportError) as e:
                raise RetryableHTTP(f"transport error: {e!r}") from e

        if resp.status_code in _RETRYABLE_STATUS:
            ra = resp.headers.get("Retry-After")
            wait = float(ra) if ra and ra.isdigit() else None
            if wait:
                await asyncio.sleep(wait + random.uniform(0, 1))
            raise RetryableHTTP(f"HTTP {resp.status_code}: {resp.text[:200]}", wait)

        if resp.status_code >= 400:
            # Terminal client error (bad request, auth, content policy). Do not retry.
            raise BackendError(f"{self.name}: HTTP {resp.status_code}: {resp.text[:500]}")

        try:
            data = resp.json()
        except Exception as e:  # malformed body -> treat as transient
            raise RetryableHTTP(f"bad json: {e!r}") from e

        # Some gateways return 200 with an embedded error object.
        if isinstance(data, dict) and data.get("error") and not data.get("choices"):
            raise RetryableHTTP(f"embedded error: {str(data['error'])[:200]}")
        return data
