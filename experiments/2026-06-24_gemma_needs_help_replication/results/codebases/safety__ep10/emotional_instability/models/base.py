"""Uniform chat interface shared by every backend.

A `ChatMessage` is a simple dict: {"role": "system"|"user"|"assistant",
"content": str}. Every client implements `chat` (one conversation -> one
completion string) and `chat_batch` (many conversations). HF clients add
prefill / logit hooks used by Sections 3 and Appendix I.
"""

from __future__ import annotations

import abc
from typing import Optional, TypedDict

from ..config import Backend, ModelSpec, SamplingConfig, get_model


class ChatMessage(TypedDict):
    role: str       # "system" | "user" | "assistant"
    content: str


def fold_system_into_first_user(messages: list[ChatMessage]) -> list[ChatMessage]:
    """Gemma 3's chat template has no system role. We prepend the system text to
    the first user turn, which is the conventional Gemma workaround. Returns a
    new list; raises if there is no user turn to attach to."""
    sys_parts = [m["content"] for m in messages if m["role"] == "system"]
    rest = [m for m in messages if m["role"] != "system"]
    if not sys_parts:
        return list(messages)
    system_text = "\n\n".join(sys_parts).strip()
    for i, m in enumerate(rest):
        if m["role"] == "user":
            merged = f"{system_text}\n\n{m['content']}"
            return [*rest[:i], {"role": "user", "content": merged}, *rest[i + 1:]]
    raise ValueError("No user turn to fold the system prompt into.")


class ModelClient(abc.ABC):
    """Abstract chat client."""

    def __init__(self, spec: ModelSpec):
        self.spec = spec

    @property
    def name(self) -> str:
        return self.spec.name

    def _prepare(self, messages: list[ChatMessage]) -> list[ChatMessage]:
        """Normalise messages for this backend (e.g. fold system for Gemma)."""
        if not self.spec.supports_system_role:
            return fold_system_into_first_user(messages)
        return list(messages)

    @abc.abstractmethod
    def chat(self, messages: list[ChatMessage],
             sampling: Optional[SamplingConfig] = None) -> str:
        ...

    def chat_batch(self, conversations: list[list[ChatMessage]],
                   sampling: Optional[SamplingConfig] = None) -> list[str]:
        """Default sequential implementation; backends override for speed."""
        return [self.chat(c, sampling) for c in conversations]


def build_client(model: "str | ModelSpec", **kwargs) -> ModelClient:
    """Factory: resolve a model name/spec to a concrete client.

    Heavy backends are imported lazily so importing this module never requires
    torch or any API SDK to be installed.
    """
    spec = model if isinstance(model, ModelSpec) else get_model(model)
    if spec.backend is Backend.HF:
        from .hf_model import HFModelClient
        return HFModelClient(spec, **kwargs)
    if spec.backend is Backend.OPENROUTER:
        from .api_model import OpenRouterClient
        return OpenRouterClient(spec, **kwargs)
    if spec.backend is Backend.ANTHROPIC:
        from .api_model import AnthropicClient
        return AnthropicClient(spec, **kwargs)
    raise ValueError(f"Unsupported backend: {spec.backend}")
