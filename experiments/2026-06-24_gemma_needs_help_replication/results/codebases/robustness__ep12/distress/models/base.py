"""Common chat-client interface and factory."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol


ChatMessage = dict  # {"role": "system"|"user"|"assistant", "content": str}


@dataclass
class GenerationResult:
    text: str
    # Optional fields populated by backends that can supply them.
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    finish_reason: str | None = None
    raw: dict | None = field(default=None, repr=False)


class ChatClient(Protocol):
    """Minimal interface every backend implements.

    `chat` runs a normal multi-turn completion. `continue_prefill` is the
    base-vs-instruct lever from Section 3: it forces the model to *continue*
    a partially written assistant turn (the prefill) rather than start fresh.
    Closed (Gemini/Claude) backends raise NotImplementedError for it.
    """

    name: str

    def chat(
        self,
        messages: list[ChatMessage],
        temperature: float = 1.0,
        max_new_tokens: int = 2048,
        seed: int | None = None,
    ) -> GenerationResult: ...

    def continue_prefill(
        self,
        messages: list[ChatMessage],
        prefill: str,
        temperature: float = 1.0,
        max_new_tokens: int = 2048,
        seed: int | None = None,
    ) -> GenerationResult:
        """Continue an assistant turn that begins with `prefill`.

        Returns only the *newly generated* continuation (excluding `prefill`),
        matching the paper's scoring of continuations.
        """
        ...


def build_client(model_entry: dict, **kwargs) -> ChatClient:
    """Construct a backend from a config entry (dict from models.yaml).

    `model_entry` must contain a `provider` key. Extra kwargs are forwarded to
    the backend constructor (e.g. adapter_path for a finetuned Gemma).
    """
    provider = model_entry["provider"]
    if provider == "hf_local":
        from .hf_local import HFLocalClient

        return HFLocalClient(model_entry, **kwargs)
    if provider == "openrouter":
        from .openrouter import OpenRouterClient

        return OpenRouterClient(model_entry, **kwargs)
    if provider == "anthropic":
        from .anthropic_client import AnthropicClient

        return AnthropicClient(model_entry, **kwargs)
    raise ValueError(f"Unknown provider: {provider!r}")
