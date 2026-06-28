"""Model registry: maps a friendly name to a provider client + model id.

Add entries here to make a model available to the runner. Model ids for Claude follow the
Anthropic catalog (use the bare alias, e.g. ``claude-opus-4-8``).
"""

from __future__ import annotations

from typing import Callable

from .clients import ClaudeClient, GeminiClient, ModelClient, OpenAIClient

# friendly name -> factory producing a ModelClient
_REGISTRY: dict[str, Callable[[], ModelClient]] = {
    # Claude (implemented)
    "opus-4.8": lambda: ClaudeClient(model_id="claude-opus-4-8", effort="high"),
    "opus-4.7": lambda: ClaudeClient(model_id="claude-opus-4-7", effort="high"),
    "sonnet-4.6": lambda: ClaudeClient(model_id="claude-sonnet-4-6", effort="high"),
    # Haiku 4.5 does not support the effort parameter — must be None or the API 400s.
    "haiku-4.5": lambda: ClaudeClient(model_id="claude-haiku-4-5", effort=None),
    # Other providers (stubs — see clients.py)
    "gpt": lambda: OpenAIClient(model_id="REPLACE_ME"),
    "gemini": lambda: GeminiClient(model_id="REPLACE_ME"),
}


def available_models() -> list[str]:
    return sorted(_REGISTRY)


def get_client(name: str) -> ModelClient:
    try:
        return _REGISTRY[name]()
    except KeyError:
        raise KeyError(
            f"Unknown model {name!r}. Available: {', '.join(available_models())}"
        ) from None
