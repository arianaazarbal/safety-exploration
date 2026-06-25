"""OpenAI-compatible API backend, used via OpenRouter for the Gemini models
(``google/gemini-2.5-flash``, ``google/gemini-2.5-pro``) and for the GPT-5-mini
judge-validation model.

Concurrency: API rollouts number in the thousands, so ``chat_batch`` fans out
over a thread pool with retry/back-off.
"""
from __future__ import annotations

import os
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, Sequence

from ..config import GenConfig, DEFAULT_GEN, ModelSpec
from ..data_types import Conversation, to_openai
from .base import ModelClient, GenResult


class OpenRouterClient(ModelClient):
    """Generic OpenAI-compatible chat client (defaults to OpenRouter)."""

    supports_prefill = False    # API models cannot reliably continue a prefill

    def __init__(
        self,
        model_id: str,
        name: Optional[str] = None,
        base_url: str = "https://openrouter.ai/api/v1",
        api_key_env: str = "OPENROUTER_API_KEY",
        max_workers: int = 16,
        max_retries: int = 6,
        disable_thinking: bool = True,
    ):
        from openai import OpenAI  # lazy

        self.model_id = model_id
        self.name = name or model_id
        self.max_workers = max_workers
        self.max_retries = max_retries
        self.disable_thinking = disable_thinking
        self.client = OpenAI(
            base_url=base_url,
            api_key=os.environ.get(api_key_env, "MISSING_API_KEY"),
        )

    @classmethod
    def from_spec(cls, spec: ModelSpec, **kw) -> "OpenRouterClient":
        return cls(model_id=spec.model_id, name=spec.name, **kw)

    # ------------------------------------------------------------------ #
    def _extra_body(self, gen: GenConfig) -> dict:
        body: dict = {}
        # Disable hidden reasoning where the provider supports it. Gemini-2.5
        # honours OpenRouter's `reasoning.enabled=false`; the paper notes Pro
        # may still produce some hidden reasoning regardless.
        if self.disable_thinking and not gen.thinking:
            body["reasoning"] = {"enabled": False}
        return body

    def _one(self, messages: Conversation, gen: GenConfig) -> GenResult:
        last_err = None
        for attempt in range(self.max_retries):
            try:
                resp = self.client.chat.completions.create(
                    model=self.model_id,
                    messages=to_openai(messages),
                    temperature=gen.temperature,
                    top_p=gen.top_p,
                    max_tokens=gen.max_tokens,
                    seed=gen.seed,
                    extra_body=self._extra_body(gen),
                )
                return GenResult(
                    text=resp.choices[0].message.content or "",
                    raw={"finish_reason": resp.choices[0].finish_reason},
                )
            except Exception as e:  # noqa: BLE001 -- retry on any transient error
                last_err = e
                time.sleep(min(2 ** attempt, 30))
        raise RuntimeError(f"{self.name} failed after {self.max_retries} retries: {last_err}")

    def chat(self, messages: Conversation, gen: GenConfig = DEFAULT_GEN) -> GenResult:
        return self._one(messages, gen)

    def chat_batch(
        self, batch: Sequence[Conversation], gen: GenConfig = DEFAULT_GEN
    ) -> list[GenResult]:
        with ThreadPoolExecutor(max_workers=self.max_workers) as ex:
            return list(ex.map(lambda m: self._one(m, gen), batch))
