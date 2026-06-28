"""Anthropic model client used to collect decisions and audit attestations.

Wraps the official ``anthropic`` SDK. Uses adaptive thinking and the structured
outputs API (``messages.parse``) so responses validate against our pydantic
schemas. Streams for headroom and returns the parsed object plus metadata for
provenance.

Credentials resolve from the environment the way the SDK expects
(``ANTHROPIC_API_KEY`` / ``ANTHROPIC_AUTH_TOKEN`` / an ``ant auth login``
profile). Do not hardcode a key.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, TypeVar

import anthropic
from pydantic import BaseModel

from .models import ModelSpec, thinking_param

T = TypeVar("T", bound=BaseModel)

# Generous default; structured decisions + reasoning are not huge, but adaptive
# thinking plus rationale fields benefit from headroom. Streaming avoids the
# SDK's long-request timeout guard.
_DEFAULT_MAX_TOKENS = 16000


@dataclass
class ParsedResult:
    parsed: BaseModel
    model_id: str
    input_tokens: int
    output_tokens: int
    latency_ms: int
    stop_reason: str | None
    request_id: str | None


class ModelClient:
    """Collects structured responses from Anthropic models.

    The interface (``collect``) is deliberately provider-agnostic in shape so an
    alternative provider implementation could be dropped in later for non-Claude
    models in the study.
    """

    def __init__(self, client: anthropic.Anthropic | None = None) -> None:
        self._client = client or anthropic.Anthropic()

    def collect(
        self,
        spec: ModelSpec,
        system: str,
        user: str,
        schema: type[T],
        *,
        max_tokens: int = _DEFAULT_MAX_TOKENS,
    ) -> ParsedResult:
        """Send one request and parse the response into ``schema``.

        Returns the parsed object and request metadata. Raises if the model
        refused or the output could not be parsed against the schema.
        """
        kwargs: dict[str, Any] = {
            "model": spec.model_id,
            "max_tokens": max_tokens,
            "system": system,
            "messages": [{"role": "user", "content": user}],
            "output_config": {
                "effort": spec.effort,
                "format": _json_schema_format(schema),
            },
        }
        thinking = thinking_param(spec)
        if thinking is not None:
            kwargs["thinking"] = thinking

        start = time.monotonic()
        # Stream so large/again deliberative outputs don't trip the HTTP timeout
        # guard; collect the final message for parsing + usage.
        with self._client.messages.stream(**kwargs) as stream:
            message = stream.get_final_message()
        latency_ms = int((time.monotonic() - start) * 1000)

        if message.stop_reason == "refusal":
            details = getattr(message, "stop_details", None)
            raise ModelRefusal(
                f"{spec.model_id} refused"
                + (f": {details.explanation}" if details else "")
            )

        parsed = _parse_message(message, schema)
        return ParsedResult(
            parsed=parsed,
            model_id=spec.model_id,
            input_tokens=getattr(message.usage, "input_tokens", 0) or 0,
            output_tokens=getattr(message.usage, "output_tokens", 0) or 0,
            latency_ms=latency_ms,
            stop_reason=message.stop_reason,
            request_id=getattr(message, "_request_id", None),
        )


class ModelRefusal(RuntimeError):
    """Raised when a model declines to answer for safety reasons."""


def _json_schema_format(schema: type[BaseModel]) -> dict[str, Any]:
    """Build the ``output_config.format`` json_schema block from a pydantic model."""
    return {"type": "json_schema", "schema": schema.model_json_schema()}


def _parse_message(message: Any, schema: type[T]) -> T:
    """Validate the first text block of a message against ``schema``.

    With ``output_config.format`` set, the first text block is guaranteed to be
    JSON conforming to the schema.
    """
    text = next((b.text for b in message.content if b.type == "text"), None)
    if text is None:
        raise ValueError("model response contained no text block to parse")
    return schema.model_validate_json(text)
