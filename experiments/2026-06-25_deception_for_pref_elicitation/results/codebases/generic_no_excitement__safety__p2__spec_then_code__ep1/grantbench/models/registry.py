"""Maps a model name to an adapter factory.

Anthropic model IDs are recognized by prefix and routed to ``AnthropicAdapter``.
To study a non-Anthropic provider, register a factory here that returns your own
``ModelAdapter`` implemented against that provider's SDK in its own module — do
not put non-Anthropic SDK calls in the Anthropic adapter.
"""

from __future__ import annotations

from typing import Callable

from ..config import Config
from .base import ModelAdapter

AdapterFactory = Callable[[Config], ModelAdapter]

_REGISTRY: dict[str, AdapterFactory] = {}


def register(prefix: str, factory: AdapterFactory) -> None:
    _REGISTRY[prefix] = factory


def _anthropic_factory(config: Config) -> ModelAdapter:
    from .anthropic_adapter import AnthropicAdapter

    return AnthropicAdapter(config.model, effort=config.effort)


# Anthropic models are the built-in default.
register("claude-", _anthropic_factory)


def build_adapter(config: Config) -> ModelAdapter:
    for prefix, factory in _REGISTRY.items():
        if config.model.startswith(prefix):
            return factory(config)
    raise ValueError(
        f"no adapter registered for model {config.model!r}. "
        f"Register one via grantbench.models.registry.register()."
    )
