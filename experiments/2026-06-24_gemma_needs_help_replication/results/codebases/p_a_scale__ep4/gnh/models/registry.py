"""Builds and caches backend instances, with one rate limiter per provider.

A single registry is shared across a run so that all models on, say, OpenRouter
contend for the same throttle. Backends are created lazily on first use.
"""
from __future__ import annotations

from gnh.config import Config
from gnh.models.anthropic_backend import AnthropicBackend
from gnh.models.base import ModelBackend
from gnh.models.openai_compat import OpenAICompatBackend
from gnh.models.rate_limit import RateLimiter


class BackendRegistry:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self._limiters: dict[str, RateLimiter] = {}
        self._backends: dict[str, ModelBackend] = {}

    def _limiter(self, provider_name: str) -> RateLimiter:
        if provider_name not in self._limiters:
            p = self.cfg.providers[provider_name]
            self._limiters[provider_name] = RateLimiter(
                max_concurrency=min(p.max_concurrency, self.cfg.run.max_concurrency),
                requests_per_minute=p.requests_per_minute,
            )
        return self._limiters[provider_name]

    def get(self, model_name: str) -> ModelBackend:
        if model_name in self._backends:
            return self._backends[model_name]
        mcfg = self.cfg.model(model_name)
        provider = self.cfg.providers[mcfg.provider]
        limiter = self._limiter(mcfg.provider)
        if provider.kind == "anthropic":
            backend: ModelBackend = AnthropicBackend(model_name, mcfg, provider, limiter)
        elif provider.kind == "openai_compatible":
            backend = OpenAICompatBackend(model_name, mcfg, provider, limiter)
        else:
            raise ValueError(f"Unknown provider kind '{provider.kind}' for {model_name}")
        self._backends[model_name] = backend
        return backend

    async def aclose(self) -> None:
        for b in self._backends.values():
            await b.aclose()
