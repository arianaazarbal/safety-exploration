"""Map a model id string to a constructed adapter.

Routing is by id prefix. Add a provider by registering its prefixes and factory.
"""

from __future__ import annotations

from typing import Any, Callable

from .base import ModelAdapter


def _anthropic_factory(model: str, **kwargs: Any) -> ModelAdapter:
    from .anthropic_adapter import AnthropicAdapter

    return AnthropicAdapter(model, **kwargs)


def _openai_factory(model: str, **kwargs: Any) -> ModelAdapter:
    from .stub_adapters import OpenAIAdapter

    return OpenAIAdapter(model, **kwargs)


def _google_factory(model: str, **kwargs: Any) -> ModelAdapter:
    from .stub_adapters import GoogleAdapter

    return GoogleAdapter(model, **kwargs)


# Prefix -> factory. First matching prefix wins.
_ROUTES: list[tuple[str, Callable[..., ModelAdapter]]] = [
    ("claude", _anthropic_factory),
    ("gpt", _openai_factory),
    ("o1", _openai_factory),
    ("o3", _openai_factory),
    ("gemini", _google_factory),
]


def build_adapter(model: str, **kwargs: Any) -> ModelAdapter:
    for prefix, factory in _ROUTES:
        if model.startswith(prefix):
            return factory(model, **kwargs)
    raise ValueError(
        f"No adapter registered for model id {model!r}. "
        f"Known prefixes: {sorted(p for p, _ in _ROUTES)}."
    )
