"""Gemini (and optional GPT cross-judge) via OpenRouter's OpenAI-compatible API.

Mirrors the paper, which accessed Gemini-2.5-{flash,pro} through OpenRouter with
"thinking" disabled. Closed API models cannot be token-prefilled, so prefill
raises PrefillNotSupported (Section 3 therefore covers Gemma only — see DESIGN.md).
"""
from __future__ import annotations

import concurrent.futures as cf
import os

from ..utils import get_logger, retry
from .base import GenConfig, Message, ModelBackend, PrefillNotSupported

log = get_logger(__name__)


class OpenRouterBackend(ModelBackend):
    def __init__(self, spec, *, max_workers: int = 8):
        super().__init__(spec)
        from openai import OpenAI

        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.environ["OPENROUTER_API_KEY"],
        )
        self.max_workers = max_workers

    def _one(self, conversation: list[Message], cfg: GenConfig) -> list[str]:
        extra = dict(self.spec.extra_body or {})

        def call():
            resp = self.client.chat.completions.create(
                model=self.spec.api_id,
                messages=conversation,
                temperature=cfg.temperature,
                top_p=cfg.top_p,
                max_tokens=cfg.max_tokens,
                n=cfg.n,
                seed=cfg.seed,
                extra_body=extra or None,
            )
            return [c.message.content or "" for c in resp.choices]

        return retry(call)

    def chat_batch(self, conversations, cfg, prefill=None):
        if prefill is not None:
            raise PrefillNotSupported(
                f"{self.name}: token-level prefill unavailable for API models"
            )
        with cf.ThreadPoolExecutor(max_workers=self.max_workers) as ex:
            return list(ex.map(lambda c: self._one(c, cfg), conversations))

    def complete_batch(self, prompts, cfg):
        # Wrap raw prompts as single user messages (best effort for API models).
        convs = [[{"role": "user", "content": p}] for p in prompts]
        return self.chat_batch(convs, cfg)
