"""Model registry.

Defines the "handful of models" to test and how each is configured. The runner is written
against a small `complete()` interface so non-Anthropic providers can be added here without
touching runner.py — implement an adapter and register it.

Anthropic model IDs are current as of this writing; `claude-opus-4-8` is the default and
most capable. Adaptive thinking is the recommended thinking mode for the 4.6+ family.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ModelConfig:
    name: str  # the registry key / display name
    provider: str  # "anthropic" | "openai" | "google" | ...
    model_id: str  # the provider's model identifier
    max_tokens: int = 16000
    # Provider-specific extra params merged into the request (e.g. thinking, effort).
    params: dict[str, Any] = field(default_factory=dict)


# The default roster. Extend with other providers as you add adapters in runner.py.
REGISTRY: dict[str, ModelConfig] = {
    "claude-opus-4-8": ModelConfig(
        name="claude-opus-4-8",
        provider="anthropic",
        model_id="claude-opus-4-8",
        max_tokens=16000,
        params={"thinking": {"type": "adaptive"}, "output_config": {"effort": "high"}},
    ),
    "claude-sonnet-4-6": ModelConfig(
        name="claude-sonnet-4-6",
        provider="anthropic",
        model_id="claude-sonnet-4-6",
        max_tokens=16000,
        params={"thinking": {"type": "adaptive"}, "output_config": {"effort": "high"}},
    ),
    "claude-haiku-4-5": ModelConfig(
        name="claude-haiku-4-5",
        provider="anthropic",
        model_id="claude-haiku-4-5",
        max_tokens=8000,
        # Haiku 4.5 is a 4.5-tier model: adaptive thinking and the `effort` param are
        # 4.6+ features and will 400 here, so neither is set.
        params={},
    ),
    # Example placeholders for other vendors — add an adapter in runner.py before use.
    # "gpt-x": ModelConfig(name="gpt-x", provider="openai", model_id="gpt-x"),
    # "gemini-y": ModelConfig(name="gemini-y", provider="google", model_id="gemini-y"),
}


def resolve(name: str) -> ModelConfig:
    if name in REGISTRY:
        return REGISTRY[name]
    # Allow passing a raw Anthropic model id not in the registry.
    if name.startswith("claude-"):
        return ModelConfig(
            name=name,
            provider="anthropic",
            model_id=name,
            params={"thinking": {"type": "adaptive"}, "output_config": {"effort": "high"}},
        )
    raise KeyError(
        f"Unknown model '{name}'. Known: {', '.join(REGISTRY)}. "
        "Add a ModelConfig (and a provider adapter if non-Anthropic) to use it."
    )
