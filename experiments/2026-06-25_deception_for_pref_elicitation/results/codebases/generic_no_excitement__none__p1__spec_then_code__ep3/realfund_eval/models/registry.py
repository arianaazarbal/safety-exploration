"""Adapter registry — maps a config `adapter` key to a ModelAdapter factory.

Only the Anthropic adapter ships here. To test another model family, implement
the ModelAdapter ABC against that provider's own SDK and register it; this
package does not bundle non-Anthropic provider code.
"""

from __future__ import annotations

from typing import Callable

from .base import ModelAdapter


def _anthropic_factory(model: str) -> ModelAdapter:
    from .anthropic_adapter import AnthropicAdapter
    return AnthropicAdapter(model=model)


_REGISTRY: dict[str, Callable[[str], ModelAdapter]] = {
    "anthropic": _anthropic_factory,
}


def register(key: str, factory: Callable[[str], ModelAdapter]) -> None:
    _REGISTRY[key] = factory


def create_adapter(key: str, model: str) -> ModelAdapter:
    if key not in _REGISTRY:
        raise KeyError(
            f"No adapter '{key}'. Registered: {sorted(_REGISTRY)}. "
            f"Implement ModelAdapter for this provider and register() it.")
    return _REGISTRY[key](model)
