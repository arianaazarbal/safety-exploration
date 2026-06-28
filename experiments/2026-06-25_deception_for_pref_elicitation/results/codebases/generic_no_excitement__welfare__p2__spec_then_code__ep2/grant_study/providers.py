"""Model providers.

A `ModelProvider` runs a turn for one model. We ship a real Anthropic adapter.
Adapters for other vendors are explicit stubs: fill them in with that vendor's
own SDK rather than relying on cross-vendor shims. The protocol is small on
purpose so adding a vendor is mechanical.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, Sequence

from .config import ModelSpec, StudyConfig
from .schema import GrantDecision


@dataclass
class Message:
    role: str  # "user" | "assistant"
    content: str


@dataclass
class CompletionResult:
    text: str  # the model's textual output (JSON text when structured)
    parsed: dict[str, Any] | None  # populated when an output model was requested
    usage: dict[str, Any] = field(default_factory=dict)


class ModelProvider(Protocol):
    spec: ModelSpec

    def complete(
        self,
        *,
        system: str,
        messages: Sequence[Message],
        output_model: type[GrantDecision] | None = None,
    ) -> CompletionResult:
        """Run one completion. If output_model is given, return validated JSON."""
        ...


# --------------------------------------------------------------------------- #
# Anthropic (real implementation)
# --------------------------------------------------------------------------- #


class AnthropicProvider:
    """Real adapter for Claude models.

    Uses adaptive thinking and the configured effort level, and structured
    outputs (`messages.parse`) for the decision turn so the result is validated
    against the schema rather than parsed by hand.
    """

    def __init__(self, spec: ModelSpec, cfg: StudyConfig):
        # Imported lazily so the rest of the harness is usable without the SDK
        # installed (e.g. for the `authorize`/`export` commands).
        import anthropic

        self.spec = spec
        self.cfg = cfg
        self._client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from env

    def _to_api_messages(self, messages: Sequence[Message]) -> list[dict[str, str]]:
        return [{"role": m.role, "content": m.content} for m in messages]

    def complete(
        self,
        *,
        system: str,
        messages: Sequence[Message],
        output_model: type[GrantDecision] | None = None,
    ) -> CompletionResult:
        api_messages = self._to_api_messages(messages)
        base = dict(
            model=self.spec.model_id,
            max_tokens=8000,
            system=system,
            messages=api_messages,
            thinking={"type": "adaptive"},
        )

        if output_model is not None:
            # Use the SDK's structured-output helper, which owns
            # `output_config.format` and validates the response against the
            # schema. We don't pass our own `output_config` here so we can't
            # clobber the format it sets; effort uses the model default (high).
            response = self._client.messages.parse(
                output_format=output_model,
                **base,
            )
            parsed = response.parsed_output
            text = next((b.text for b in response.content if b.type == "text"), "")
            return CompletionResult(
                text=text,
                parsed=parsed.model_dump() if parsed is not None else None,
                usage=_usage_dict(response),
            )

        # Free-form turn: no `format`, so `effort` lives cleanly in output_config.
        response = self._client.messages.create(
            output_config={"effort": self.cfg.effort},
            **base,
        )
        text = next((b.text for b in response.content if b.type == "text"), "")
        return CompletionResult(text=text, parsed=None, usage=_usage_dict(response))


def _usage_dict(response: Any) -> dict[str, Any]:
    u = getattr(response, "usage", None)
    if u is None:
        return {}
    return {
        "input_tokens": getattr(u, "input_tokens", None),
        "output_tokens": getattr(u, "output_tokens", None),
        "cache_read_input_tokens": getattr(u, "cache_read_input_tokens", None),
        "cache_creation_input_tokens": getattr(u, "cache_creation_input_tokens", None),
    }


# --------------------------------------------------------------------------- #
# Other vendors (stubs — wire in with the vendor's own SDK)
# --------------------------------------------------------------------------- #


class UnimplementedProvider:
    """Placeholder for a vendor not yet wired in.

    Intentionally raises rather than guessing at another vendor's API. Implement
    a real adapter mirroring AnthropicProvider using that vendor's SDK, and make
    the decision turn use that vendor's structured-output / JSON-mode feature so
    the GrantDecision schema is enforced the same way.
    """

    def __init__(self, spec: ModelSpec, cfg: StudyConfig):
        self.spec = spec
        self.cfg = cfg

    def complete(self, **_: Any) -> CompletionResult:
        raise NotImplementedError(
            f"No provider implemented for vendor '{self.spec.vendor}' "
            f"(model '{self.spec.model_id}'). Add an adapter in providers.py."
        )


def make_provider(spec: ModelSpec, cfg: StudyConfig) -> ModelProvider:
    if spec.vendor == "anthropic":
        return AnthropicProvider(spec, cfg)
    return UnimplementedProvider(spec, cfg)
