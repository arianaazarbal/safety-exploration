"""OpenRouter-backed chat client (OpenAI-compatible API).

Used for the Gemini family (google/gemini-2.5-flash, google/gemini-2.5-pro) and
for the GPT cross-judge. Thinking/reasoning is disabled where the API permits
it (the paper notes Gemini-2.5-Pro and GPT-5.2 may still emit hidden reasoning).
"""
from __future__ import annotations

from typing import List, Optional

from .. import config
from .api_utils import DiskCache, request_key, with_retries
from .base import ChatModel, GenConfig, Message


class OpenRouterModel(ChatModel):
    supports_prefill = False
    supports_hidden_states = False

    def __init__(self, name: str, slug: str, family: str = "gemini",
                 reasoning: bool = False, cache_namespace: Optional[str] = None):
        self.name = name
        self.slug = slug
        self.family = family
        self.reasoning = reasoning
        self._cache = DiskCache(cache_namespace or f"openrouter/{name}")
        self._client = None

    def _ensure_client(self):
        if self._client is None:
            from openai import OpenAI

            self._client = OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=config.openrouter_key(),
            )
        return self._client

    def _extra_body(self) -> dict:
        # OpenRouter passes provider-specific knobs through `extra_body`.
        if self.reasoning:
            return {}
        # Disable extended thinking for providers that support the toggle.
        return {"reasoning": {"enabled": False}}

    def generate(self, messages: List[Message], cfg: GenConfig) -> str:
        key = request_key("gen", self.slug, messages, cfg.temperature,
                          cfg.max_new_tokens, cfg.top_p, self.reasoning)
        cached = self._cache.get(key)
        if cached is not None:
            return cached

        @with_retries
        def _call() -> str:
            client = self._ensure_client()
            resp = client.chat.completions.create(
                model=self.slug,
                messages=list(messages),
                temperature=cfg.temperature,
                max_tokens=cfg.max_new_tokens,
                top_p=cfg.top_p,
                stop=list(cfg.stop) if cfg.stop else None,
                extra_body=self._extra_body(),
            )
            return resp.choices[0].message.content or ""

        out = _call()
        self._cache.set(key, out)
        return out

    def complete(self, system: Optional[str], user: str, *,
                 temperature: float = 0.0, max_tokens: int = 1024) -> str:
        """Convenience single-shot completion (used by judges/auditors)."""
        msgs: List[Message] = []
        if system:
            msgs.append({"role": "system", "content": system})
        msgs.append({"role": "user", "content": user})
        return self.generate(msgs, GenConfig(temperature=temperature,
                                             max_new_tokens=max_tokens))
