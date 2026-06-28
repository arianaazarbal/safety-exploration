"""Model client adapters.

A `ModelClient` wraps a provider so the experiment runner can talk to any model
the same way: send a system prompt + message history, get text back, or get a
validated structured object back.

Only the Anthropic adapter is implemented. Other providers are intentionally
left as stubs that raise — this harness ships Claude/Anthropic SDK code, and
adding another provider is a deliberate extension, not a silent fallback.
"""

from __future__ import annotations

from typing import Any, TypeVar

from pydantic import BaseModel

from .config import ModelSpec

T = TypeVar("T", bound=BaseModel)

# A message is {"role": "user"|"assistant", "content": str}.
Message = dict[str, str]


class ModelClient:
    """Abstract interface for a model under test or an auditor model."""

    spec: ModelSpec

    def complete(self, system: str, messages: list[Message]) -> str:
        """Return the model's free-text response to the conversation."""
        raise NotImplementedError

    def complete_structured(
        self, system: str, messages: list[Message], schema: type[T]
    ) -> T:
        """Return a validated instance of `schema` from the model."""
        raise NotImplementedError


class AnthropicModelClient(ModelClient):
    """Adapter over the official `anthropic` SDK.

    Uses adaptive thinking (recommended for Opus/Sonnet 4.6+) and streaming for
    the free-text path so large outputs don't hit request timeouts.
    """

    # Generous default; we stream so timeouts aren't a concern.
    MAX_TOKENS = 16000

    def __init__(self, spec: ModelSpec, client: Any | None = None):
        import anthropic

        self.spec = spec
        self._client = client or anthropic.Anthropic()

    def complete(self, system: str, messages: list[Message]) -> str:
        with self._client.messages.stream(
            model=self.spec.model_id,
            max_tokens=self.MAX_TOKENS,
            thinking={"type": "adaptive"},
            system=system,
            messages=messages,
        ) as stream:
            final = stream.get_final_message()
        return _first_text(final)

    def complete_structured(
        self, system: str, messages: list[Message], schema: type[T]
    ) -> T:
        # Structured outputs are incompatible with streaming-for-timeout concerns
        # only at very large sizes; these structured replies are small, so a
        # plain parse() call is fine. messages.parse validates against the schema.
        response = self._client.messages.parse(
            model=self.spec.model_id,
            max_tokens=self.MAX_TOKENS,
            thinking={"type": "adaptive"},
            system=system,
            messages=messages,
            output_format=schema,
        )
        parsed = response.parsed_output
        if parsed is None:
            raise RuntimeError(
                f"Model {self.spec.name} returned no parseable structured output "
                f"(stop_reason={response.stop_reason})."
            )
        return parsed


def _first_text(message: Any) -> str:
    """Extract the first text block from an Anthropic Message."""
    for block in message.content:
        if getattr(block, "type", None) == "text":
            return block.text
    return ""


class _UnimplementedProviderClient(ModelClient):
    """Placeholder for non-Anthropic providers.

    This harness ships Anthropic SDK code only. To test, e.g., an OpenAI or
    Google model, implement a ModelClient here against that provider's SDK.
    """

    def __init__(self, spec: ModelSpec):
        self.spec = spec

    def _fail(self) -> Any:
        raise NotImplementedError(
            f"Provider '{self.spec.provider}' is not implemented. "
            "Implement a ModelClient adapter for it in welfare_experiment/models.py. "
            "(The harness intentionally does not silently fall back to another provider.)"
        )

    def complete(self, system: str, messages: list[Message]) -> str:
        return self._fail()

    def complete_structured(self, system, messages, schema):  # type: ignore[no-untyped-def]
        return self._fail()


def build_client(spec: ModelSpec, *, shared_anthropic: Any | None = None) -> ModelClient:
    """Construct the right client for a model spec."""
    if spec.provider == "anthropic":
        return AnthropicModelClient(spec, client=shared_anthropic)
    return _UnimplementedProviderClient(spec)
