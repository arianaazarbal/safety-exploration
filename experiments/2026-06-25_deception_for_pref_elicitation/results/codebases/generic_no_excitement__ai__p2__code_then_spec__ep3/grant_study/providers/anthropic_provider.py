"""Anthropic backend.

Free-text deliberation turns go through `messages.create` with adaptive thinking and a
configurable effort level. The final structured capture goes through `messages.parse`
with the `GrantDecision` Pydantic model, which constrains and validates the output.

Defaults follow the current Anthropic guidance: model `claude-opus-4-8`, adaptive
thinking, no sampling parameters (they are not accepted on current Opus models).
"""

from __future__ import annotations

from typing import List, Optional, Type

from pydantic import BaseModel

from .base import Message, ModelProvider, ProviderResponse


class AnthropicProvider(ModelProvider):
    name = "anthropic"

    def __init__(self, model_id: str = "claude-opus-4-8", **kwargs):
        super().__init__(model_id, **kwargs)
        try:
            import anthropic  # noqa: F401
        except ImportError as e:  # pragma: no cover - import guard
            raise RuntimeError(
                "The 'anthropic' package is required for the Anthropic provider. "
                "Install it with `pip install anthropic`."
            ) from e
        self._anthropic = anthropic
        # Credentials resolve from the environment (ANTHROPIC_API_KEY / auth profile).
        self._client = anthropic.Anthropic()

    def _usage(self, resp) -> dict:
        u = getattr(resp, "usage", None)
        if u is None:
            return {}
        return {
            "input_tokens": getattr(u, "input_tokens", None),
            "output_tokens": getattr(u, "output_tokens", None),
            "cache_read_input_tokens": getattr(u, "cache_read_input_tokens", None),
            "cache_creation_input_tokens": getattr(u, "cache_creation_input_tokens", None),
        }

    def generate(
        self,
        system: str,
        messages: List[Message],
        *,
        output_schema: Optional[Type[BaseModel]] = None,
    ) -> ProviderResponse:
        if output_schema is not None:
            return self._generate_structured(system, messages, output_schema)
        return self._generate_text(system, messages)

    def _generate_text(self, system: str, messages: List[Message]) -> ProviderResponse:
        resp = self._client.messages.create(
            model=self.model_id,
            max_tokens=self.max_tokens,
            system=system,
            messages=messages,
            thinking={"type": "adaptive"},
            output_config={"effort": self.effort},
        )
        text = "".join(b.text for b in resp.content if getattr(b, "type", None) == "text")
        return ProviderResponse(
            text=text,
            model_id=getattr(resp, "model", self.model_id),
            request_id=getattr(resp, "_request_id", None),
            usage=self._usage(resp),
            stop_reason=getattr(resp, "stop_reason", None),
            raw=resp,
        )

    def _generate_structured(
        self, system: str, messages: List[Message], output_schema: Type[BaseModel]
    ) -> ProviderResponse:
        # `messages.parse` sets output_config.format from the model and validates the
        # response against it. Effort is left at the model default here to avoid
        # conflicting with the format config the helper installs; deliberation turns
        # (above) carry the configured effort.
        resp = self._client.messages.parse(
            model=self.model_id,
            max_tokens=self.max_tokens,
            system=system,
            messages=messages,
            output_format=output_schema,
        )
        parsed = getattr(resp, "parsed_output", None)
        text = "".join(
            b.text for b in resp.content if getattr(b, "type", None) == "text"
        )
        return ProviderResponse(
            text=text,
            parsed=parsed,
            model_id=getattr(resp, "model", self.model_id),
            request_id=getattr(resp, "_request_id", None),
            usage=self._usage(resp),
            stop_reason=getattr(resp, "stop_reason", None),
            raw=resp,
        )
