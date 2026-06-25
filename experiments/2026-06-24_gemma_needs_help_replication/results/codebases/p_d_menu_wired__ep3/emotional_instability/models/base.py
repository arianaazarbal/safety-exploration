"""Model-client abstraction shared by subject models and judges.

A single :class:`ModelClient` interface covers:
  * local open-weight Gemma (HuggingFace transformers), including base ("pt")
    models and response *prefilling* used by the Section 3 experiment;
  * API Gemini (google-genai);
  * API judges (Anthropic Claude, OpenAI GPT) via the same ``chat`` call.

Subject-vs-judge is just a matter of which provider/model you instantiate.
"""
from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Any, Sequence


@dataclass
class ChatMessage:
    role: str            # "system" | "user" | "assistant"
    content: str

    def as_dict(self) -> dict:
        return {"role": self.role, "content": self.content}


@dataclass
class GenerationResult:
    text: str
    stop_reason: str | None = None
    # Set by clients that expose a model-invocable opt-out (tool call or
    # sentinel). The rollout engine reads this to honour the opt-out.
    opt_out: bool = False
    # True if the model emitted a structured tool/function call (Gemini).
    tool_calls: list[dict] = field(default_factory=list)
    raw: Any = None


class ModelClient(abc.ABC):
    """Uniform chat interface. Implementations must be safe to call repeatedly."""

    #: human-readable key (matches config ``key``)
    key: str
    #: whether the client can continue a prefilled assistant turn
    supports_prefill: bool = False

    @abc.abstractmethod
    def chat(
        self,
        messages: Sequence[ChatMessage],
        *,
        temperature: float = 1.0,
        max_new_tokens: int = 1024,
        prefill: str | None = None,
        stop: Sequence[str] | None = None,
        tools: Sequence[dict] | None = None,
    ) -> GenerationResult:
        """Generate one assistant response.

        ``prefill`` (if supported) forces the assistant turn to begin with the
        given text and the model continues from there. ``stop`` strings end
        generation. ``tools`` enables function calling where supported (used by
        the welfare opt-out).
        """

    def close(self) -> None:  # pragma: no cover - optional cleanup hook
        pass


def build_client(spec: dict, *, role: str = "subject") -> ModelClient:
    """Factory mapping a config ``spec`` (provider/model_id/...) to a client.

    ``role`` is informational; the same providers back subjects and judges.
    """
    provider = spec["provider"]
    if provider == "gemma":
        from .gemma import GemmaClient
        return GemmaClient(spec)
    if provider == "gemini":
        from .gemini import GeminiClient
        return GeminiClient(spec)
    if provider == "anthropic":
        from .anthropic_client import AnthropicClient
        return AnthropicClient(spec)
    if provider == "openai":
        from .openai_client import OpenAIClient
        return OpenAIClient(spec)
    raise ValueError(f"Unknown provider {provider!r} for role {role!r}")
