"""Async OpenRouter chat-completions client for the target models.

We speak the OpenAI-compatible /chat/completions endpoint that OpenRouter
exposes. Concurrency is bounded by a shared semaphore; transient failures
(timeouts, 429, 5xx) are retried with exponential backoff + jitter.
"""

from __future__ import annotations

import asyncio
import os
import random
from dataclasses import dataclass
from typing import List, Optional

import httpx

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"


@dataclass
class ChatMessage:
    role: str   # "user" | "assistant" | "system"
    content: str

    def to_dict(self) -> dict:
        return {"role": self.role, "content": self.content}


class GenerationError(RuntimeError):
    """Raised when a generation fails after all retries."""


class OpenRouterClient:
    def __init__(
        self,
        api_key: Optional[str] = None,
        max_concurrency: int = 8,
        max_retries: int = 5,
        timeout: float = 120.0,
        seed: int = 0,
    ):
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
        if not self.api_key:
            raise RuntimeError("OPENROUTER_API_KEY is not set (see .env.example).")
        self.max_retries = max_retries
        self._sem = asyncio.Semaphore(max_concurrency)
        self._rng = random.Random(seed)
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        # Optional OpenRouter attribution headers.
        if os.environ.get("OPENROUTER_APP_URL"):
            headers["HTTP-Referer"] = os.environ["OPENROUTER_APP_URL"]
        if os.environ.get("OPENROUTER_APP_TITLE"):
            headers["X-Title"] = os.environ["OPENROUTER_APP_TITLE"]
        self._client = httpx.AsyncClient(headers=headers, timeout=timeout)

    async def aclose(self) -> None:
        await self._client.aclose()

    async def __aenter__(self) -> "OpenRouterClient":
        return self

    async def __aexit__(self, *exc) -> None:
        await self.aclose()

    async def generate(
        self,
        model_id: str,
        messages: List[ChatMessage],
        temperature: float,
        max_tokens: int,
        disable_reasoning: bool = False,
    ) -> str:
        """Return the assistant's text for a single completion."""
        payload = {
            "model": model_id,
            "messages": [m.to_dict() for m in messages],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if disable_reasoning:
            # OpenRouter unified flag to suppress hidden reasoning where the
            # provider supports it (mirrors the paper's thinking=false). For
            # Gemini this disables thinking; ignored by models without it.
            payload["reasoning"] = {"enabled": False}

        async with self._sem:
            return await self._post_with_retries(payload, model_id)

    async def _post_with_retries(self, payload: dict, model_id: str) -> str:
        last_err: Optional[Exception] = None
        for attempt in range(self.max_retries):
            try:
                resp = await self._client.post(OPENROUTER_URL, json=payload)
                if resp.status_code in (429, 500, 502, 503, 504):
                    raise _RetryableStatus(resp.status_code, resp.text[:300])
                resp.raise_for_status()
                data = resp.json()
                return self._extract_text(data, model_id)
            except (_RetryableStatus, httpx.TransportError, httpx.TimeoutException) as err:
                last_err = err
                await asyncio.sleep(self._backoff(attempt))
            except httpx.HTTPStatusError as err:
                # Non-retryable HTTP error (e.g. 400/401/403): surface immediately.
                raise GenerationError(
                    f"{model_id}: HTTP {err.response.status_code}: {err.response.text[:300]}"
                ) from err
        raise GenerationError(f"{model_id}: failed after {self.max_retries} retries: {last_err}")

    @staticmethod
    def _extract_text(data: dict, model_id: str) -> str:
        try:
            choice = data["choices"][0]
            msg = choice["message"]
            content = msg.get("content")
            if isinstance(content, list):
                # Some providers return content as a list of parts.
                content = "".join(
                    part.get("text", "") for part in content if isinstance(part, dict)
                )
            if content is None:
                content = ""
            return content
        except (KeyError, IndexError, TypeError) as err:
            raise GenerationError(f"{model_id}: unexpected response shape: {data}") from err

    def _backoff(self, attempt: int) -> float:
        # Exponential backoff with full jitter, capped.
        base = min(2.0 ** attempt, 30.0)
        return self._rng.uniform(0, base)


class _RetryableStatus(Exception):
    def __init__(self, status: int, body: str):
        super().__init__(f"retryable status {status}: {body}")
        self.status = status
