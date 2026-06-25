"""OpenRouter (OpenAI-compatible) backend for Gemini targets and the GPT-5-mini
cross-check judge.

The paper accesses all API models through OpenRouter (Appendix B.1) and sets
thinking to false where supported.  Gemini is closed-source, so it supports
neither prefilling nor internal-state access -- those methods raise, matching
the paper's stated limitation that interventions and base-model studies cannot
be run on Gemini.
"""
from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor

from ..config import ModelConfig
from .base import ChatModel, GenerationOptions, Message


class OpenRouterChatModel(ChatModel):
    def __init__(self, cfg: ModelConfig):
        super().__init__(cfg)
        from openai import OpenAI

        api_key = os.environ.get(cfg.api_key_env or "OPENROUTER_API_KEY")
        if not api_key:
            raise RuntimeError(
                f"Missing API key: set ${cfg.api_key_env or 'OPENROUTER_API_KEY'}"
            )
        self.client = OpenAI(
            base_url=cfg.api_base or "https://openrouter.ai/api/v1",
            api_key=api_key,
        )
        self._max_concurrency = int(cfg.extra.get("max_concurrency", 8))

    def _extra_body(self) -> dict:
        body: dict = {}
        if self.cfg.disable_thinking:
            # OpenRouter normalises provider reasoning controls under "reasoning".
            # Gemini-2.5 Pro / GPT-5 may still emit hidden reasoning (App. B.1).
            body["reasoning"] = {"enabled": False}
        body.update(self.cfg.extra.get("extra_body", {}))
        return body

    def _one(self, conversation: list[Message], opts: GenerationOptions) -> str:
        resp = self.client.chat.completions.create(
            model=self.cfg.model_id,
            messages=conversation,
            temperature=opts.temperature,
            max_tokens=opts.max_new_tokens,
            top_p=opts.top_p,
            stop=opts.stop,
            seed=opts.seed,
            extra_body=self._extra_body() or None,
        )
        return resp.choices[0].message.content or ""

    def generate_batch(
        self, conversations: list[list[Message]], opts: GenerationOptions | None = None
    ) -> list[str]:
        o = self._resolved(opts)
        if len(conversations) == 1:
            return [self._one(conversations[0], o)]
        results: list[str] = [""] * len(conversations)
        with ThreadPoolExecutor(max_workers=self._max_concurrency) as pool:
            futures = {
                pool.submit(self._one, conv, o): i for i, conv in enumerate(conversations)
            }
            for fut in futures:
                results[futures[fut]] = fut.result()
        return results
