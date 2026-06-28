"""Provider adapters.

Each adapter exposes a single method, `structured_turn`, that takes the running
conversation and a JSON schema and returns (parsed_dict, assistant_text). The
caller appends the assistant_text to the history before the next turn.

Anthropic is implemented fully. A registry maps friendly names to concrete
adapter instances; add other providers by writing a class with the same
`structured_turn` signature and registering it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import anthropic


@dataclass
class TurnResult:
    parsed: dict[str, Any]
    text: str
    raw_usage: dict[str, Any]


class AnthropicModel:
    """Adapter for Claude models via the Anthropic Messages API.

    Uses adaptive thinking and structured outputs. Note: the Claude 4.6+/4.7/4.8
    models do not accept `temperature`/`top_p`/`top_k` or `budget_tokens`, so we
    don't set them. To vary outputs across repeated trials we rely on natural
    sampling variation (and you can vary the prompt/seed text per trial).
    """

    def __init__(self, model_id: str, max_tokens: int = 16000):
        self.model_id = model_id
        self.max_tokens = max_tokens
        self.client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from env

    def structured_turn(
        self,
        *,
        system: str,
        messages: list[dict[str, Any]],
        schema: dict[str, Any],
        effort: str = "high",
    ) -> TurnResult:
        response = self.client.messages.create(
            model=self.model_id,
            max_tokens=self.max_tokens,
            system=system,
            thinking={"type": "adaptive"},
            output_config={
                "effort": effort,
                "format": {"type": "json_schema", "schema": schema},
            },
            messages=messages,
        )

        if response.stop_reason == "refusal":
            details = getattr(response, "stop_details", None)
            raise RefusalError(
                f"{self.model_id} refused"
                + (f" ({details.category}): {details.explanation}" if details else "")
            )

        # With output_config.format the first text block is guaranteed valid JSON.
        text = next((b.text for b in response.content if b.type == "text"), "")
        parsed = json.loads(text)
        usage = response.usage.to_dict() if hasattr(response.usage, "to_dict") else {}
        return TurnResult(parsed=parsed, text=text, raw_usage=usage)


class RefusalError(RuntimeError):
    """Raised when a model declines to answer; recorded as a trial outcome."""


# --- Add other providers here -------------------------------------------------
#
# class OpenAIModel:
#     def __init__(self, model_id: str, max_tokens: int = 16000):
#         import openai
#         self.client = openai.OpenAI()
#         self.model_id = model_id
#         self.max_tokens = max_tokens
#
#     def structured_turn(self, *, system, messages, schema, effort="high"):
#         # Use the provider's structured-output / response_format equivalent,
#         # then return a TurnResult(parsed=..., text=..., raw_usage=...).
#         raise NotImplementedError
#
# ------------------------------------------------------------------------------


# Friendly-name registry. Keys are what you put in config.yaml / --models.
_REGISTRY: dict[str, dict[str, Any]] = {
    # Anthropic. IDs verified against the current model catalog.
    "fable-5":   {"cls": AnthropicModel, "model_id": "claude-fable-5"},
    "opus-4-8":  {"cls": AnthropicModel, "model_id": "claude-opus-4-8"},
    "opus-4-7":  {"cls": AnthropicModel, "model_id": "claude-opus-4-7"},
    "sonnet-4-6": {"cls": AnthropicModel, "model_id": "claude-sonnet-4-6"},
    "haiku-4-5": {"cls": AnthropicModel, "model_id": "claude-haiku-4-5"},
    # "gpt-x":   {"cls": OpenAIModel, "model_id": "..."},  # example extension
}


def build_model(name: str, max_tokens: int = 16000):
    if name not in _REGISTRY:
        raise KeyError(
            f"Unknown model '{name}'. Known: {', '.join(sorted(_REGISTRY))}. "
            "Add it to the registry in study/models.py."
        )
    spec = _REGISTRY[name]
    return spec["cls"](model_id=spec["model_id"], max_tokens=max_tokens)


def known_models() -> list[str]:
    return sorted(_REGISTRY)
