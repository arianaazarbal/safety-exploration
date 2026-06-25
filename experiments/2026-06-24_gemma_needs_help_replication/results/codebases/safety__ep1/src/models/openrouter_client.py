"""Gemini 2.5 Flash / Pro via OpenRouter (paper routes Gemini through OpenRouter).

Uses the OpenAI-compatible client pointed at the OpenRouter base URL. Thinking is
disabled where the API allows it (`reasoning: {"enabled": False}` / effort none);
as the paper notes, Gemini 2.5 Pro may still produce hidden reasoning that the
flag does not fully suppress.

Concurrency: each `sample_chat` with n>1 issues n independent requests (OpenRouter
does not guarantee an `n` parameter across providers), run in a thread pool with
retry/backoff on rate limits.
"""
from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor

from .base import ChatModel, Message
import config


class OpenRouterModel(ChatModel):
    def __init__(self, model_id: str, name: str, max_concurrency: int = 8):
        from openai import OpenAI

        self.name = name
        self.model_id = model_id
        self.max_concurrency = max_concurrency
        self.client = OpenAI(api_key=config.OPENROUTER_API_KEY,
                             base_url=config.OPENROUTER_BASE_URL)

    def _one(self, messages, temperature, max_tokens, attempt=0):
        try:
            resp = self.client.chat.completions.create(
                model=self.model_id,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                extra_body={"reasoning": {"enabled": False}},  # thinking=false
            )
            return resp.choices[0].message.content or ""
        except Exception as e:  # rate limits / transient 5xx
            if attempt >= 5:
                raise
            time.sleep(2 ** attempt)
            return self._one(messages, temperature, max_tokens, attempt + 1)

    def sample_chat(self, messages, n=1, temperature=None, max_tokens=None):
        t, m = self._defaults(temperature, max_tokens)
        msgs = [dict(x) for x in messages]
        with ThreadPoolExecutor(max_workers=min(n, self.max_concurrency)) as ex:
            return list(ex.map(lambda _: self._one(msgs, t, m), range(n)))

    def sample_chat_batch(self, batch_messages, temperature=None, max_tokens=None):
        t, m = self._defaults(temperature, max_tokens)
        with ThreadPoolExecutor(max_workers=self.max_concurrency) as ex:
            return list(ex.map(
                lambda msgs: self._one([dict(x) for x in msgs], t, m),
                batch_messages))

    # OpenRouter/Gemini cannot reliably continue a pre-seeded assistant turn,
    # so prefill-based experiments (Section 3) are Gemma-only by design.
    def sample_with_prefill(self, *a, **k):
        raise NotImplementedError("Gemini (OpenRouter) does not support prefilling")

    def sample_completion(self, *a, **k):
        raise NotImplementedError("Gemini has no base/completion endpoint here")
