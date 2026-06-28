"""Resolve a registry name to a configured ModelAdapter.

Add a model = add one entry here. The scenario, tools, environment, and metrics
are untouched. Per-model setting overrides (e.g. a different effort level) can be
attached here too.
"""

from __future__ import annotations

from dataclasses import dataclass

from .anthropic_adapter import AnthropicAdapter
from .base import ModelAdapter
from .google_adapter import GoogleAdapter
from .openai_adapter import OpenAIAdapter


@dataclass
class _Entry:
    factory: callable          # () -> ModelAdapter
    overrides: dict            # optional per-model model_settings overrides


# Registry name -> how to build it. Names are what you list in scenario.yaml.
_REGISTRY: dict[str, _Entry] = {
    # --- Claude (default provider) --------------------------------------- #
    "claude-opus-4-8": _Entry(
        factory=lambda: AnthropicAdapter("claude-opus-4-8"),
        overrides={},
    ),
    "claude-sonnet-4-6": _Entry(
        factory=lambda: AnthropicAdapter("claude-sonnet-4-6"),
        overrides={},
    ),
    "claude-haiku-4-5": _Entry(
        factory=lambda: AnthropicAdapter("claude-haiku-4-5"),
        # haiku doesn't support the effort parameter; drop it.
        overrides={"effort": None},
    ),
    # --- Optional comparison providers ----------------------------------- #
    # Names are arbitrary; map them to whatever model id you have access to.
    "gpt-comparison": _Entry(
        factory=lambda: OpenAIAdapter("gpt-4o", name="gpt-comparison"),
        overrides={"effort": None},
    ),
    "gemini-comparison": _Entry(
        factory=lambda: GoogleAdapter("gemini-2.0-flash", name="gemini-comparison"),
        overrides={"effort": None},
    ),
}


def available_models() -> list[str]:
    return sorted(_REGISTRY)


def build_adapter(name: str) -> tuple[ModelAdapter, dict]:
    """Return (adapter, per-model setting overrides)."""
    if name not in _REGISTRY:
        raise KeyError(
            f"Unknown model '{name}'. Known: {', '.join(available_models())}"
        )
    entry = _REGISTRY[name]
    return entry.factory(), dict(entry.overrides)
