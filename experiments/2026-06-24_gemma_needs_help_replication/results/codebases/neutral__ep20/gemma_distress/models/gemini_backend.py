"""Gemini backend via OpenRouter (OpenAI-compatible Chat Completions API).

The paper accesses Gemini-2.5-Flash/Pro through OpenRouter with thinking
disabled. We mirror that: an OpenAI-compatible client pointed at OpenRouter,
with provider-side reasoning turned off where the API allows it.

Gemini does not support assistant *prefill* via this API; ``GenRequest.prefill``
is therefore not used for Gemini (the prefill experiment, Section 3, is
Gemma-only because Gemini has no public base model anyway).
"""

from __future__ import annotations

import os
import time

import config

from .base import ChatModel, GenRequest, GenResult, Message


class GeminiOpenRouter(ChatModel):
    def __init__(self, name: str, openrouter_id: str, max_retries: int = 5):
        super().__init__(name)
        from openai import OpenAI

        api_key = os.environ.get(config.OPENROUTER_API_KEY_ENV)
        if not api_key:
            raise RuntimeError(
                f"Set {config.OPENROUTER_API_KEY_ENV} to query Gemini via OpenRouter."
            )
        self.client = OpenAI(base_url=config.OPENROUTER_BASE_URL, api_key=api_key)
        self.model_id = openrouter_id
        self.max_retries = max_retries

    def _extra_body(self) -> dict:
        # OpenRouter passes provider-specific knobs through `extra_body`.
        # Disable Gemini "thinking" (paper: thinking=false). Pro may still emit
        # hidden reasoning the API can't suppress, as the paper notes.
        if not config.GEMINI_DISABLE_THINKING:
            return {}
        return {"reasoning": {"max_tokens": 0, "exclude": True}}

    def generate(self, req: GenRequest) -> GenResult:
        if req.prefill:
            # Not supported on Gemini; fold the prefill into a final assistant
            # message is non-standard, so we just ignore it and warn once.
            pass
        last_err: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                resp = self.client.chat.completions.create(
                    model=self.model_id,
                    messages=req.messages,  # type: ignore[arg-type]
                    temperature=req.temperature,
                    top_p=req.top_p,
                    max_tokens=req.max_new_tokens,
                    extra_body=self._extra_body(),
                )
                text = (resp.choices[0].message.content or "").strip()
                return GenResult(text=text, prompt=req,
                                 meta={"model_id": self.model_id})
            except Exception as e:  # pragma: no cover - network dependent
                last_err = e
                time.sleep(min(2 ** attempt, 30))
        raise RuntimeError(f"Gemini generation failed after retries: {last_err!r}")

    def generate_batch(self, reqs: list[GenRequest]) -> list[GenResult]:
        # API-bound: parallelise across the batch with a thread pool so the
        # lockstep multi-turn rollout gets concurrency for free.
        from gemma_distress.utils.concurrency import thread_map

        if len(reqs) <= 1:
            return [self.generate(r) for r in reqs]
        return thread_map(self.generate, reqs, workers=config.API_CONCURRENCY,
                          desc=f"{self.name} gen")


def load_gemini(name: str, openrouter_id: str) -> ChatModel:
    return GeminiOpenRouter(name, openrouter_id)
