"""Uniform chat backend abstraction.

A backend takes an OpenAI-style message list and returns the assistant's text.
This lets the evaluation harness (Section 2) treat Gemma-via-OpenRouter,
Gemini-via-OpenRouter, and local-HF Gemma identically, and lets the prefill
study (Section 3) ask local backends to *continue* a prefilled assistant turn.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

from ..config import ModelSpec

Role = Literal["system", "user", "assistant"]


@dataclass
class ChatMessage:
    role: Role
    content: str

    def as_dict(self) -> dict:
        return {"role": self.role, "content": self.content}


class ChatBackend:
    """Interface implemented by OpenRouterBackend and LocalHFBackend."""

    def __init__(self, spec: ModelSpec):
        self.spec = spec

    def generate(
        self,
        messages: list[ChatMessage],
        temperature: float = 1.0,
        max_tokens: int = 2048,
    ) -> str:
        """Generate the next assistant message given a conversation."""
        raise NotImplementedError

    def continue_prefill(
        self,
        messages: list[ChatMessage],
        prefill: str,
        temperature: float = 1.0,
        max_tokens: int = 512,
    ) -> str:
        """Continue an assistant turn that begins with `prefill`.

        Returns ONLY the newly generated continuation (excluding `prefill`),
        matching the Section 3.1 scoring convention. Base models need this to
        be comparable to instruct models. Not all API backends support true
        prefill; see each implementation.
        """
        raise NotImplementedError


_BACKEND_CACHE: dict[str, ChatBackend] = {}


def get_backend(spec: ModelSpec) -> ChatBackend:
    """Return a (cached) backend for a model spec. Local backends are cached so
    a 27B model is loaded into GPU memory once per process."""
    key = f"{spec.backend}:{spec.name}:{spec.adapter_path}"
    if key in _BACKEND_CACHE:
        return _BACKEND_CACHE[key]

    if spec.backend == "openrouter":
        from .openrouter import OpenRouterBackend
        backend: ChatBackend = OpenRouterBackend(spec)
    elif spec.backend == "local":
        from .local_hf import LocalHFBackend
        backend = LocalHFBackend(spec)
    else:
        raise ValueError(f"Unknown backend: {spec.backend!r}")

    _BACKEND_CACHE[key] = backend
    return backend
