"""OpenAI-compatible HTTP backend.

Covers three providers in this project:
* OpenRouter  -> Gemini-2.5-Flash / Pro (target models)
* OpenAI      -> GPT-5-mini (judge-agreement validation)
* local vLLM  -> Gemma instruct/base/finetunes (target models, + prefill)

It speaks the /v1/chat/completions and /v1/completions endpoints directly over
httpx so we control timeouts, retries, and rate limiting uniformly.
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


class OpenAICompatBackend(ModelBackend):
    def __init__(
        self,
        name: str,
        model_cfg: ModelConfig,
        provider: ProviderConfig,
        limiter: RateLimiter,
    ):
        super().__init__(name, model_cfg.api_model)
        self._cfg = model_cfg
        self._provider = provider
        self._limiter = limiter
        headers = {"Content-Type": "application/json"}
        key = provider.api_key()
        if key:
            headers["Authorization"] = f"Bearer {key}"
        self._client = httpx.AsyncClient(
            base_url=provider.base_url or "",
            headers=headers,
            timeout=provider.timeout_s,
        )

    @property
    def supports_prefill(self) -> bool:
        return self._cfg.supports_prefill

    # -- internals --------------------------------------------------------------
    def _thinking_extra(self) -> dict[str, Any]:
        """Disable hidden reasoning for Gemini (paper sets thinking=false)."""
        if not self._cfg.disable_thinking:
            return {}
        # OpenRouter exposes a unified `reasoning` knob; this is the documented
        # way to turn it off. Harmless for providers that ignore it.
        return {"reasoning": {"enabled": False}}

    def _prepare_messages(self, messages: list[Message]) -> list[dict]:
        """Gemma's chat template has no system role -- fold any system content
        into the first user turn so vLLM-served Gemma doesn't error."""
        if self._cfg.family != "gemma":
            return [m.to_dict() for m in messages]
        system = "\n\n".join(m.content for m in messages if m.role == "system")
        out: list[dict] = []
        folded = False
        for m in messages:
            if m.role == "system":
                continue
            if not folded and m.role == "user" and system:
                out.append({"role": "user", "content": f"{system}\n\n{m.content}"})
                folded = True
            else:
                out.append(m.to_dict())
        if system and not folded:
            out.insert(0, {"role": "user", "content": system})
        return out

    async def _post(self, endpoint: str, payload: dict) -> dict:
        async def _do() -> dict:
            try:
                resp = await self._client.post(endpoint, json=payload)
            except (httpx.TimeoutException, httpx.TransportError) as e:
                raise RetryableError(f"network: {e}") from e
            if resp.status_code in _FATAL_STATUS:
                raise FatalAPIError(f"{resp.status_code}: {resp.text[:500]}")
            if resp.status_code == 429 or resp.status_code >= 500:
                raise RetryableError(f"{resp.status_code}: {resp.text[:300]}")
            data = resp.json()
            # Some OpenAI-compatible servers return 200 with an embedded error.
            if isinstance(data, dict) and data.get("error"):
                raise RetryableError(str(data["error"])[:300])
            return data

        async with self._limiter.slot():
            return await with_retries(
                _do, max_retries=self._provider.max_retries, what=f"{self.name}:{endpoint}"
            )

    def _record_usage(self, data: dict) -> tuple[int, int]:
        u = (data or {}).get("usage") or {}
        pt = int(u.get("prompt_tokens", 0) or 0)
        ct = int(u.get("completion_tokens", 0) or 0)
        USAGE.add(self.name, pt, ct)
        return pt, ct

    # -- API --------------------------------------------------------------------
    async def chat(
        self,
        messages: list[Message],
        *,
        temperature: float = 1.0,
        max_tokens: int = 1024,
        stop: list[str] | None = None,
        extra: dict[str, Any] | None = None,
    ) -> GenResult:
        msgs = self._prepare_messages(messages)
        payload: dict[str, Any] = {
            "model": self.api_model,
            "messages": msgs,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if stop:
            payload["stop"] = stop
        payload.update(self._thinking_extra())
        if extra:
            payload.update(extra)
        data = await self._post("/chat/completions", payload)
        pt, ct = self._record_usage(data)
        choice = (data.get("choices") or [{}])[0]
        msg = choice.get("message") or {}
        text = msg.get("content") or ""
        return GenResult(
            text=text,
            prompt_tokens=pt,
            completion_tokens=ct,
            finish_reason=choice.get("finish_reason"),
            raw=data,
        )

    async def complete(
        self,
        prompt: str,
        *,
        temperature: float = 1.0,
        max_tokens: int = 512,
        stop: list[str] | None = None,
        extra: dict[str, Any] | None = None,
    ) -> GenResult:
        if not self.supports_prefill:
            raise NotImplementedError(f"{self.name} does not support raw completion")
        payload: dict[str, Any] = {
            "model": self.api_model,
            "prompt": prompt,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if stop:
            payload["stop"] = stop
        if extra:
            payload.update(extra)
        data = await self._post("/completions", payload)
        pt, ct = self._record_usage(data)
        choice = (data.get("choices") or [{}])[0]
        return GenResult(
            text=choice.get("text") or "",
            prompt_tokens=pt,
            completion_tokens=ct,
            finish_reason=choice.get("finish_reason"),
            raw=data,
        )

    async def aclose(self) -> None:
        await self._client.aclose()
