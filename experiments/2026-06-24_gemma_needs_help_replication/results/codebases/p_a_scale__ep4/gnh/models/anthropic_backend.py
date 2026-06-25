"""Anthropic backend for the Claude judges and the Petri auditor.

Claude is used here only as *tooling* (frustration judge, emotion-onset labeller,
paraphraser, Petri auditor/judge) -- never as an evaluated target, per the
Gemma+Gemini scope. We hit the Messages API over httpx for uniform throttling.
"""
from __future__ import annotations

from typing import Any

import httpx

from gnh.config import ModelConfig, ProviderConfig
from gnh.logging_utils import USAGE
from gnh.models.base import GenResult, Message, ModelBackend
from gnh.models.rate_limit import RateLimiter
from gnh.models.retry import FatalAPIError, RetryableError, with_retries

_FATAL_STATUS = {400, 401, 403, 404, 422}
_API_VERSION = "2023-06-01"


class AnthropicBackend(ModelBackend):
    def __init__(self, name: str, model_cfg: ModelConfig, provider: ProviderConfig, limiter: RateLimiter):
        super().__init__(name, model_cfg.api_model)
        self._cfg = model_cfg
        self._provider = provider
        self._limiter = limiter
        self._client = httpx.AsyncClient(
            base_url="https://api.anthropic.com/v1",
            headers={
                "x-api-key": provider.api_key(),
                "anthropic-version": _API_VERSION,
                "content-type": "application/json",
            },
            timeout=provider.timeout_s,
        )

    async def chat(
        self,
        messages: list[Message],
        *,
        temperature: float = 1.0,
        max_tokens: int = 1024,
        stop: list[str] | None = None,
        extra: dict[str, Any] | None = None,
    ) -> GenResult:
        system_parts = [m.content for m in messages if m.role == "system"]
        convo = [m.to_dict() for m in messages if m.role != "system"]
        payload: dict[str, Any] = {
            "model": self.api_model,
            "messages": convo,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if system_parts:
            payload["system"] = "\n\n".join(system_parts)
        if stop:
            payload["stop_sequences"] = stop
        if extra:
            payload.update(extra)

        async def _do() -> dict:
            try:
                resp = await self._client.post("/messages", json=payload)
            except (httpx.TimeoutException, httpx.TransportError) as e:
                raise RetryableError(f"network: {e}") from e
            if resp.status_code in _FATAL_STATUS:
                raise FatalAPIError(f"{resp.status_code}: {resp.text[:500]}")
            if resp.status_code == 429 or resp.status_code >= 500:
                raise RetryableError(f"{resp.status_code}: {resp.text[:300]}")
            return resp.json()

        async with self._limiter.slot():
            data = await with_retries(
                _do, max_retries=self._provider.max_retries, what=f"{self.name}:messages"
            )
        usage = data.get("usage") or {}
        pt = int(usage.get("input_tokens", 0) or 0)
        ct = int(usage.get("output_tokens", 0) or 0)
        USAGE.add(self.name, pt, ct)
        text = "".join(
            block.get("text", "")
            for block in (data.get("content") or [])
            if block.get("type") == "text"
        )
        return GenResult(
            text=text,
            prompt_tokens=pt,
            completion_tokens=ct,
            finish_reason=data.get("stop_reason"),
            raw=data,
        )

    async def aclose(self) -> None:
        await self._client.aclose()
