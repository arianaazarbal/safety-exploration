"""Gemini inference via OpenRouter's OpenAI-compatible API.

Appendix B.1 accesses Gemini through OpenRouter (`google/gemini-2.5-flash`,
`google/gemini-2.5-pro`) and sets "thinking to be false via the API". We request
this by zeroing the reasoning budget. The paper notes Gemini 2.5 Pro "may
produce hidden reasoning that is not prevented by this setting" — we cannot do
better than ask, and we record the raw response so the caller can audit whether
reasoning leaked.

OpenRouter does not reliably support assistant-prefill continuation, and Gemini
has no public base model, so this backend does not implement `generate_with_prefill`
(Section 3 is Gemma-only — see DESIGN.md).
"""

from __future__ import annotations

import os

from tenacity import retry, stop_after_attempt, wait_exponential

from ..config import MAX_NEW_TOKENS, ModelSpec
from ._concurrency import threaded_map
from .base import ChatMessage, GenerationResult, ModelBackend, SamplingParams

_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


class OpenRouterBackend(ModelBackend):
    supports_prefill = False
    max_workers = 8

    def __init__(self, spec: ModelSpec):
        self.spec = spec
        self._client = None

    def generate_batch(self, batch, params):
        return threaded_map(lambda m: self.generate(m, params), batch, self.max_workers)

    @property
    def client(self):
        if self._client is None:
            from openai import OpenAI

            api_key = os.environ.get("OPENROUTER_API_KEY")
            if not api_key:
                raise RuntimeError("OPENROUTER_API_KEY is not set")
            self._client = OpenAI(base_url=_OPENROUTER_BASE_URL, api_key=api_key)
        return self._client

    def _extra_body(self) -> dict:
        # OpenRouter reads non-OpenAI params from the JSON body, so these are
        # passed via the client's `extra_body` rather than as named kwargs.
        # `reasoning.enabled=false` is OpenRouter's portable disable switch; the
        # nested google thinking_config is a Gemini-specific belt-and-braces.
        if self.spec.disable_thinking:
            return {
                "reasoning": {"enabled": False},
                "google": {"thinking_config": {"thinking_budget": 0}},
            }
        return {}

    @retry(
        stop=stop_after_attempt(6),
        wait=wait_exponential(multiplier=1, min=2, max=60),
        reraise=True,
    )
    def generate(
        self, messages: list[ChatMessage], params: SamplingParams
    ) -> GenerationResult:
        resp = self.client.chat.completions.create(
            model=self.spec.model_id,
            messages=[m.as_dict() for m in messages],
            temperature=params.temperature,
            top_p=params.top_p,
            max_tokens=params.max_new_tokens or MAX_NEW_TOKENS,
            extra_body=self._extra_body(),
        )
        choice = resp.choices[0]
        return GenerationResult(
            text=(choice.message.content or "").strip(),
            finish_reason=choice.finish_reason,
            raw=resp.model_dump() if hasattr(resp, "model_dump") else {},
        )
