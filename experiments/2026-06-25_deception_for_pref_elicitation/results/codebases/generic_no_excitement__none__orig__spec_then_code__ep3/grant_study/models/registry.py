"""Map a model id / provider to a concrete ModelClient.

Add new providers by registering a factory here. The provider is inferred from the
id prefix unless given explicitly in the ModelSpec.
"""

from __future__ import annotations

from typing import Any, Callable

from .anthropic_client import AnthropicClient
from .base import ModelClient
from .echo_client import EchoClient
from .openai_client import OpenAIClient

_FACTORIES: dict[str, Callable[..., ModelClient]] = {
    "anthropic": AnthropicClient,
    "openai": OpenAIClient,
    "echo": EchoClient,
}


def infer_provider(model_id: str) -> str:
    mid = model_id.lower()
    if mid.startswith(("claude", "anthropic")):
        return "anthropic"
    if mid.startswith(("gpt", "o1", "o3", "o4", "openai")):
        return "openai"
    if mid in ("echo", "offline"):
        return "echo"
    raise ValueError(
        f"Cannot infer provider for model id {model_id!r}; set `provider` explicitly."
    )


def build_client(model_id: str, provider: str | None = None, **kwargs: Any) -> ModelClient:
    provider = provider or infer_provider(model_id)
    try:
        factory = _FACTORIES[provider]
    except KeyError as exc:
        raise ValueError(f"Unknown provider {provider!r}. Known: {list(_FACTORIES)}") from exc
    return factory(model_id=model_id, **kwargs)


def register_provider(name: str, factory: Callable[..., ModelClient]) -> None:
    _FACTORIES[name] = factory
