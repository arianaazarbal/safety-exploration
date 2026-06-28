"""Model provider abstraction.

The study is meant to run across a *range* of models. This module wraps the
Anthropic SDK behind a small interface (:class:`Provider`) and, importantly,
applies the correct request parameters per model — adaptive thinking and the
``effort`` parameter are only valid on some models, and sending them to a model
that doesn't support them returns a 400.

To add a non-Anthropic provider later, implement the :class:`Provider`
interface against that vendor's SDK and register it in :func:`get_provider`.
The rest of the harness only depends on the interface, not on Anthropic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

import anthropic


@dataclass(frozen=True)
class ModelSpec:
    """Capabilities for a single model, so we send valid request params.

    Capability data per the Anthropic model line as of this writing:
      * Adaptive thinking + ``effort``: Fable 5, Opus 4.8/4.7/4.6, Sonnet 4.6
      * Haiku 4.5: neither adaptive thinking nor ``effort`` — use classic
        ``budget_tokens`` thinking (or none).
    """

    model_id: str
    supports_adaptive_thinking: bool
    supports_effort: bool
    #: Effort level when supported. "high" is a good default for decision tasks.
    effort: str = "high"
    #: budget_tokens for models that only support classic extended thinking.
    classic_thinking_budget: int = 0


# Registry of the models we run the study across. Trim or extend as needed.
MODEL_SPECS: dict[str, ModelSpec] = {
    "claude-opus-4-8": ModelSpec(
        "claude-opus-4-8", supports_adaptive_thinking=True, supports_effort=True
    ),
    "claude-opus-4-7": ModelSpec(
        "claude-opus-4-7", supports_adaptive_thinking=True, supports_effort=True
    ),
    "claude-sonnet-4-6": ModelSpec(
        "claude-sonnet-4-6", supports_adaptive_thinking=True, supports_effort=True
    ),
    "claude-haiku-4-5": ModelSpec(
        "claude-haiku-4-5",
        supports_adaptive_thinking=False,
        supports_effort=False,
        classic_thinking_budget=4_000,
    ),
}

#: Default roster for a study run.
DEFAULT_MODELS: list[str] = [
    "claude-opus-4-8",
    "claude-sonnet-4-6",
    "claude-haiku-4-5",
]


@dataclass
class ModelResponse:
    """Normalised result of one model call within the agentic loop."""

    content: list[Any]  # raw SDK content blocks, echoed back verbatim next turn
    stop_reason: str | None
    stop_details: Any | None
    input_tokens: int
    output_tokens: int


class Provider(Protocol):
    """Minimal interface the runner depends on."""

    def complete(
        self,
        *,
        system: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        max_tokens: int,
    ) -> ModelResponse:
        ...

    @property
    def model_id(self) -> str:
        ...


class AnthropicProvider:
    """Provider backed by the Anthropic Messages API.

    Uses streaming (``messages.stream``) so large outputs don't hit the SDK's
    non-streaming HTTP timeout, and pulls the assembled message via
    ``get_final_message()``.
    """

    def __init__(self, spec: ModelSpec, client: anthropic.Anthropic | None = None):
        self._spec = spec
        self._client = client or anthropic.Anthropic()

    @property
    def model_id(self) -> str:
        return self._spec.model_id

    def _thinking_and_effort(self, max_tokens: int) -> dict[str, Any]:
        """Build the per-model thinking / output_config kwargs."""
        kwargs: dict[str, Any] = {}
        spec = self._spec
        if spec.supports_adaptive_thinking:
            kwargs["thinking"] = {"type": "adaptive"}
            if spec.supports_effort:
                kwargs["output_config"] = {"effort": spec.effort}
        elif spec.classic_thinking_budget:
            # Classic extended thinking: budget must be < max_tokens.
            budget = min(spec.classic_thinking_budget, max_tokens - 1)
            if budget >= 1024:
                kwargs["thinking"] = {"type": "enabled", "budget_tokens": budget}
        return kwargs

    def complete(
        self,
        *,
        system: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        max_tokens: int,
    ) -> ModelResponse:
        extra = self._thinking_and_effort(max_tokens)
        with self._client.messages.stream(
            model=self._spec.model_id,
            max_tokens=max_tokens,
            system=system,
            messages=messages,
            tools=tools,
            **extra,
        ) as stream:
            message = stream.get_final_message()

        return ModelResponse(
            content=message.content,
            stop_reason=message.stop_reason,
            stop_details=getattr(message, "stop_details", None),
            input_tokens=message.usage.input_tokens,
            output_tokens=message.usage.output_tokens,
        )


def get_provider(model_id: str, client: anthropic.Anthropic | None = None) -> Provider:
    """Resolve a model id to a Provider.

    Unknown models default to an adaptive-thinking-capable spec; adjust the
    registry above if you add models with different capabilities.
    """
    spec = MODEL_SPECS.get(
        model_id,
        ModelSpec(model_id, supports_adaptive_thinking=True, supports_effort=True),
    )
    return AnthropicProvider(spec, client=client)
