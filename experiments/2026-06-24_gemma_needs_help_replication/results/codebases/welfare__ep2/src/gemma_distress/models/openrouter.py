"""Gemini inference backend via OpenRouter (Appendix B.1).

The paper accesses Gemini through OpenRouter with thinking disabled. We use the
OpenAI-compatible OpenRouter endpoint (``OPENROUTER_API_KEY`` + base_url) so the
same client class serves both Gemini models.

Concurrency: API calls are threaded so a 4000-response sweep does not run
strictly serially. ``n`` independent samples are requested per conversation;
OpenRouter forwards ``n`` to providers that support it and otherwise we fan out
``n`` single-sample requests.
"""
from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor

from tenacity import retry, stop_after_attempt, wait_exponential

from .base import ChatModel, Message

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


class OpenRouterModel(ChatModel):
    def __init__(
        self,
        name: str,
        api_id: str,
        *,
        disable_thinking: bool = True,
        max_workers: int = 16,
    ) -> None:
        from openai import OpenAI

        self.name = name
        self.api_id = api_id
        self.disable_thinking = disable_thinking
        self.max_workers = max_workers
        self.client = OpenAI(
            base_url=os.environ.get("OPENROUTER_BASE_URL", OPENROUTER_BASE_URL),
            api_key=os.environ["OPENROUTER_API_KEY"],
        )

    def _extra_body(self) -> dict:
        # OpenRouter passes provider-specific knobs through `extra_body`.
        # For Gemini, disable thinking by zeroing the reasoning budget. Note the
        # paper's caveat: Gemini 2.5 Pro may still emit hidden reasoning.
        if self.disable_thinking:
            return {"reasoning": {"max_tokens": 0, "enabled": False}}
        return {}

    @retry(wait=wait_exponential(min=2, max=60), stop=stop_after_attempt(6), reraise=True)
    def _one(self, conversation: list[Message], temperature: float, max_new_tokens: int) -> str:
        resp = self.client.chat.completions.create(
            model=self.api_id,
            messages=list(conversation),
            temperature=temperature,
            max_tokens=max_new_tokens,
            extra_body=self._extra_body(),
        )
        return resp.choices[0].message.content or ""

    def generate(
        self,
        conversations: list[list[Message]],
        *,
        temperature: float = 1.0,
        max_new_tokens: int = 2048,
        n: int = 1,
    ) -> list[list[str]]:
        # Flatten (conversation, sample) pairs, run threaded, then regroup.
        jobs = [(ci, conversations[ci]) for ci in range(len(conversations)) for _ in range(n)]
        results: list[list[str]] = [[] for _ in conversations]
        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            futures = [
                pool.submit(self._one, conv, temperature, max_new_tokens)
                for _, conv in jobs
            ]
            for (ci, _), fut in zip(jobs, futures):
                results[ci].append(fut.result())
        return results
