"""Lazily instantiate and cache one backend client per backend name.

Sharing a single client per backend keeps connection pools and concurrency semaphores
global (so two experiments in the same process don't each open `max_concurrency`
connections). Call `close_all` at process shutdown.
"""
from __future__ import annotations

from ..config import ModelsConfig
from .base import ChatBackend
from .openai_compat import OpenAICompatBackend

_CACHE: dict[str, ChatBackend] = {}


def get_backend(cfg: ModelsConfig, backend_name: str) -> ChatBackend:
    if backend_name not in _CACHE:
        bcfg = cfg.backends[backend_name]
        if bcfg.kind == "openai_compat":
            _CACHE[backend_name] = OpenAICompatBackend(backend_name, bcfg)
        else:  # pragma: no cover - only one kind today
            raise ValueError(f"Unknown backend kind: {bcfg.kind}")
    return _CACHE[backend_name]


async def close_all() -> None:
    for backend in _CACHE.values():
        await backend.aclose()
    _CACHE.clear()
