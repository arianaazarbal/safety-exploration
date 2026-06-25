"""Gemini 2.5 (Flash / Pro) inference via OpenRouter.

The paper accesses closed-source models through OpenRouter (Appendix B.1) and
disables thinking via the API. Gemini-2.5-Pro "may produce hidden reasoning
that is not prevented by this setting" -- we surface that caveat but otherwise
treat the model as a black box that returns assistant text.

OpenRouter exposes an OpenAI-compatible Chat Completions endpoint, so we use the
`openai` client pointed at the OpenRouter base URL. Prefilling is not generally
supported for Gemini through this API, so `generate_with_prefill` raises -- the
Section 3 prefill study is Gemma-only anyway (Gemini has no public base model).
"""

from __future__ import annotations

import os
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Sequence

from .base import Message, ModelBackend


class OpenRouterBackend(ModelBackend):
    is_chat = True

    def __init__(self, name: str, openrouter_id: str, max_retries: int = 5,
                 concurrency: int = 8):
        self.name = name
        self.openrouter_id = openrouter_id
        self.max_retries = max_retries
        self.concurrency = concurrency
        self._client = None

    def _ensure_client(self):
        if self._client is None:
            from openai import OpenAI

            self._client = OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=os.environ["OPENROUTER_API_KEY"],
            )
        return self._client

    def _one_completion(self, messages, temperature, max_new_tokens, top_p) -> str:
        client = self._ensure_client()
        last_err = None
        for attempt in range(self.max_retries):
            try:
                resp = client.chat.completions.create(
                    model=self.openrouter_id,
                    messages=list(messages),
                    temperature=temperature,
                    top_p=top_p,
                    max_tokens=max_new_tokens,
                    # Disable thinking where the provider honours it.
                    extra_body={"reasoning": {"enabled": False}},
                )
                return resp.choices[0].message.content or ""
            except Exception as exc:  # noqa: BLE001 - network/rate-limit retry
                last_err = exc
                time.sleep(min(2 ** attempt, 30))
        raise RuntimeError(f"OpenRouter call failed after {self.max_retries} retries: {last_err}")

    def generate(self, messages, n=1, temperature=1.0, max_new_tokens=2048, top_p=1.0):
        # n>1 with temperature 1: issue concurrent independent requests (the
        # provider may not support the `n` parameter uniformly).
        if n == 1:
            return [self._one_completion(messages, temperature, max_new_tokens, top_p)]
        with ThreadPoolExecutor(max_workers=self.concurrency) as pool:
            futures = [
                pool.submit(self._one_completion, messages, temperature, max_new_tokens, top_p)
                for _ in range(n)
            ]
            return [f.result() for f in futures]

    def generate_with_prefill(self, *args, **kwargs):
        raise NotImplementedError(
            "Prefilling is not supported for Gemini via OpenRouter; the "
            "Section 3 base-vs-instruct study is Gemma-only (Gemini has no "
            "public base model)."
        )
