"""Async LLM client wrapper over the OpenRouter (OpenAI-compatible) API.

Handles concurrency limiting, retries with exponential backoff, and best-effort
suppression of model-side reasoning. Used for both target generations and judge
scoring.
"""
from __future__ import annotations

import asyncio
from typing import Dict, List, Optional

from openai import AsyncOpenAI

from config import Config


class LLMClient:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self._client = AsyncOpenAI(
            base_url=cfg.api_base,
            api_key=cfg.api_key,
            timeout=cfg.request_timeout_seconds,
            default_headers={
                # Optional OpenRouter attribution headers.
                "HTTP-Referer": "https://github.com/replication/gemma-distress",
                "X-Title": "gemma-distress-replication",
            },
        )
        self._sem = asyncio.Semaphore(cfg.concurrency)

    async def complete(
        self,
        model: str,
        messages: List[Dict[str, str]],
        temperature: float,
        max_tokens: int,
        disable_reasoning: bool = False,
    ) -> Optional[str]:
        """Return the assistant message content, or None on persistent failure."""
        extra_body: Dict = {}
        if disable_reasoning:
            # OpenRouter's unified reasoning control; ignored by models that
            # don't support it. For Gemini this maps to disabling thinking.
            extra_body["reasoning"] = {"enabled": False}

        last_err: Optional[Exception] = None
        async with self._sem:
            for attempt in range(self.cfg.max_retries):
                try:
                    resp = await self._client.chat.completions.create(
                        model=model,
                        messages=messages,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        extra_body=extra_body or None,
                    )
                    if not resp.choices:
                        raise RuntimeError("empty choices in response")
                    content = resp.choices[0].message.content
                    if content is None:
                        raise RuntimeError("null content in response")
                    return content
                except Exception as e:  # broad: network, rate-limit, 5xx, parse
                    last_err = e
                    if attempt < self.cfg.max_retries - 1:
                        delay = self.cfg.backoff_base_seconds * (2 ** attempt)
                        await asyncio.sleep(delay)
        print(f"[client] giving up on {model} after "
              f"{self.cfg.max_retries} attempts: {last_err}")
        return None

    async def aclose(self) -> None:
        await self._client.close()
