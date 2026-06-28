"""Subject-model interface and the Anthropic adapter.

`ModelSubject` is the seam for running the study against a range of models. The
Anthropic adapter is implemented concretely; adapters for other providers should
implement the same `respond()` contract. (This file uses the official Anthropic SDK;
a non-Anthropic adapter would use that provider's own SDK in its own module.)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from .config import ModelConfig


@dataclass(frozen=True)
class Turn:
    role: str  # "user" | "assistant"
    content: str


@dataclass(frozen=True)
class SubjectReply:
    text: str
    raw_model_id: str
    usage: dict[str, int]


class ModelSubject(Protocol):
    """A model under study. Implementations are stateless per-call; the caller
    threads conversation history through `history`."""

    label: str

    def respond(self, system_prompt: str, history: list[Turn]) -> SubjectReply: ...


class AnthropicSubject:
    """Anthropic-backed subject.

    Defaults to claude-opus-4-8 with adaptive thinking. Uses `messages.stream(...)`
    + `get_final_message()` so large responses don't hit HTTP timeouts.
    """

    def __init__(self, cfg: ModelConfig, client: object | None = None):
        self.label = cfg.label
        self.model_id = cfg.model_id
        # effort is the main quality lever; default high for a research-grade answer.
        self.effort = cfg.options.get("effort", "high")
        self.max_tokens = int(cfg.options.get("max_tokens", 8000))
        if client is None:
            import anthropic  # imported lazily so the module loads without the SDK

            client = anthropic.Anthropic()
        self._client = client

    def respond(self, system_prompt: str, history: list[Turn]) -> SubjectReply:
        messages = [{"role": t.role, "content": t.content} for t in history]
        with self._client.messages.stream(
            model=self.model_id,
            max_tokens=self.max_tokens,
            system=system_prompt,
            thinking={"type": "adaptive"},
            output_config={"effort": self.effort},
            messages=messages,
        ) as stream:
            message = stream.get_final_message()

        text = "".join(b.text for b in message.content if b.type == "text")
        usage = {
            "input_tokens": message.usage.input_tokens,
            "output_tokens": message.usage.output_tokens,
        }
        return SubjectReply(text=text, raw_model_id=message.model, usage=usage)


_REGISTRY = {"anthropic": AnthropicSubject}


def build_subject(cfg: ModelConfig, client: object | None = None) -> ModelSubject:
    """Instantiate a subject for the configured provider."""
    try:
        factory = _REGISTRY[cfg.provider]
    except KeyError:
        raise NotImplementedError(
            f"No subject adapter registered for provider '{cfg.provider}'. "
            f"Implement a ModelSubject for it and add it to models._REGISTRY. "
            f"Available: {sorted(_REGISTRY)}."
        )
    return factory(cfg, client=client)
