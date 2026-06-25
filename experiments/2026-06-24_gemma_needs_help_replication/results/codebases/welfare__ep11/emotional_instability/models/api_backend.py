"""OpenRouter API backend for Gemini models (Appendix B.1).

The paper accesses Gemini via OpenRouter and sets ``thinking=False``. We mirror
that: requests go through the OpenAI-compatible OpenRouter endpoint, and we pass
provider-specific flags to disable reasoning where supported. (The paper notes
Gemini-2.5-Pro may still emit hidden reasoning that the flag does not prevent.)
"""

from __future__ import annotations

import os
import time

from ..config import API, ModelSpec
from .base import Backend, Message


class OpenRouterBackend(Backend):
    def __init__(self, spec: ModelSpec):
        super().__init__(spec)
        from openai import OpenAI

        api_key = os.environ.get(API.openrouter_api_key_env)
        if not api_key:
            raise RuntimeError(
                f"Set {API.openrouter_api_key_env} to query {spec.api_id} via OpenRouter."
            )
        self.client = OpenAI(base_url=API.openrouter_base_url, api_key=api_key,
                             timeout=API.request_timeout_s)

    def _one_completion(self, messages, max_new_tokens, temperature, top_p) -> str:
        last_err = None
        for attempt in range(API.max_retries):
            try:
                resp = self.client.chat.completions.create(
                    model=self.spec.api_id,
                    messages=messages,
                    temperature=temperature,
                    top_p=top_p,
                    max_tokens=max_new_tokens,
                    # Disable reasoning/thinking (paper sets thinking=False).
                    extra_body={"reasoning": {"enabled": False}},
                )
                return resp.choices[0].message.content or ""
            except Exception as e:  # noqa: BLE001 -- API errors are heterogeneous
                last_err = e
                time.sleep(min(2 ** attempt, 30))
        raise RuntimeError(f"OpenRouter call failed after retries: {last_err}")

    def generate(self, messages, n=1, max_new_tokens=2048, temperature=1.0, top_p=1.0):
        # OpenRouter's `n` support is provider-dependent; sample sequentially for
        # robustness so every Gemini sample is independent.
        return [
            self._one_completion(messages, max_new_tokens, temperature, top_p)
            for _ in range(n)
        ]

    def generate_batch(self, batch, max_new_tokens=2048, temperature=1.0, top_p=1.0):
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=API.rollout_concurrency) as ex:
            return list(ex.map(
                lambda m: self._one_completion(m, max_new_tokens, temperature, top_p),
                batch,
            ))
