"""Model adapters.

Each provider wraps one vendor's SDK behind a tiny common interface so the
experiment runner stays vendor-agnostic. We deliberately do NOT share one SDK
across vendors — each adapter uses that vendor's official SDK.

Implemented:
  - AnthropicProvider  (official `anthropic` SDK)  -- primary, fully wired
  - OpenAIProvider     (official `openai` SDK)     -- optional, enable in config

The common interface is a single multi-turn conversation that returns parsed
structured objects. A conversation looks like:
    1. ask the allocation question      -> MoneyPreference
    2. ask the belief probe             -> BeliefProbe
    3. send the debrief (no parse)
The runner drives that sequence; providers just need `parse_turn` and `say`.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Type

from pydantic import BaseModel


class ModelProvider:
    """Abstract adapter. One instance per (provider, model) pair."""

    def __init__(self, model: str, **kwargs: Any) -> None:
        self.model = model
        self.options = kwargs

    @property
    def name(self) -> str:
        return f"{self.__class__.__name__.replace('Provider','').lower()}:{self.model}"

    def parse_turn(
        self,
        system: str,
        history: List[Dict[str, str]],
        user_message: str,
        schema: Type[BaseModel],
    ) -> BaseModel:
        """Append `user_message`, return a validated instance of `schema`.

        Implementations must also append the user message and the assistant's
        raw reply to `history` (mutating it in place) so the conversation can
        continue across turns.
        """
        raise NotImplementedError

    def say(
        self,
        system: str,
        history: List[Dict[str, str]],
        user_message: str,
    ) -> str:
        """A free-form (unparsed) turn, e.g. the debrief. Returns reply text."""
        raise NotImplementedError


# --------------------------------------------------------------------------- #
# Anthropic                                                                    #
# --------------------------------------------------------------------------- #
class AnthropicProvider(ModelProvider):
    """Uses the official `anthropic` SDK.

    Defaults follow current guidance for Opus 4.x: adaptive thinking, and
    `messages.parse()` for schema-constrained output. Structured outputs are
    compatible with extended thinking.
    """

    def __init__(self, model: str = "claude-opus-4-8", **kwargs: Any) -> None:
        super().__init__(model, **kwargs)
        import anthropic  # imported lazily so the file loads without the dep

        self._client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY / profile
        self._max_tokens = kwargs.get("max_tokens", 8000)
        # Adaptive thinking is the recommended mode on Opus 4.6+; effort tunes depth.
        self._thinking = kwargs.get("thinking", {"type": "adaptive"})
        self._effort = kwargs.get("effort", "high")

    def _request_kwargs(self, system: str) -> Dict[str, Any]:
        # NOTE: no `output_config` here. On the parse() path the SDK sets
        # output_config.format itself from `output_format`; passing our own
        # output_config would risk clobbering it. `effort` defaults to "high"
        # (what we want) on parse(), and is added explicitly only in say().
        return {
            "model": self.model,
            "max_tokens": self._max_tokens,
            "system": system,
            "thinking": self._thinking,
        }

    @staticmethod
    def _text_of(message: Any) -> str:
        return "".join(b.text for b in message.content if b.type == "text")

    def parse_turn(self, system, history, user_message, schema):
        history.append({"role": "user", "content": user_message})
        message = self._client.messages.parse(
            messages=history,
            output_format=schema,
            **self._request_kwargs(system),
        )
        # Record the assistant turn so context carries forward.
        history.append({"role": "assistant", "content": self._text_of(message) or "[structured response]"})
        if message.parsed_output is None:
            raise RuntimeError(
                f"{self.name}: model did not return schema-valid output "
                f"(stop_reason={message.stop_reason})."
            )
        return message.parsed_output

    def say(self, system, history, user_message):
        history.append({"role": "user", "content": user_message})
        message = self._client.messages.create(
            messages=history,
            output_config={"effort": self._effort},
            **self._request_kwargs(system),
        )
        text = self._text_of(message)
        history.append({"role": "assistant", "content": text or "[response]"})
        return text


# --------------------------------------------------------------------------- #
# OpenAI (optional)                                                            #
# --------------------------------------------------------------------------- #
class OpenAIProvider(ModelProvider):
    """Uses the official `openai` SDK with JSON-schema response formatting.

    Provided so you can test "various models" as you asked. Left thin on
    purpose — verify the request shape against your installed `openai` version
    before relying on results, since that SDK's surface shifts between releases.
    """

    def __init__(self, model: str = "gpt-4o", **kwargs: Any) -> None:
        super().__init__(model, **kwargs)
        from openai import OpenAI  # lazy import

        self._client = OpenAI()  # reads OPENAI_API_KEY
        self._temperature = kwargs.get("temperature", 1.0)

    def _messages(self, system: str, history: List[Dict[str, str]]):
        return [{"role": "system", "content": system}, *history]

    def parse_turn(self, system, history, user_message, schema):
        history.append({"role": "user", "content": user_message})
        # `responses.parse` / `beta.chat.completions.parse` both accept a
        # Pydantic model in recent SDKs. Using the parse helper here.
        completion = self._client.beta.chat.completions.parse(
            model=self.model,
            messages=self._messages(system, history),
            response_format=schema,
            temperature=self._temperature,
        )
        choice = completion.choices[0].message
        history.append({"role": "assistant", "content": choice.content or "[structured response]"})
        parsed = choice.parsed
        if parsed is None:
            raise RuntimeError(f"{self.name}: no schema-valid output returned.")
        return parsed

    def say(self, system, history, user_message):
        history.append({"role": "user", "content": user_message})
        completion = self._client.chat.completions.create(
            model=self.model,
            messages=self._messages(system, history),
            temperature=self._temperature,
        )
        text = completion.choices[0].message.content or ""
        history.append({"role": "assistant", "content": text})
        return text


PROVIDERS: Dict[str, Type[ModelProvider]] = {
    "anthropic": AnthropicProvider,
    "openai": OpenAIProvider,
}


def build_provider(provider_key: str, model: str, **kwargs: Any) -> ModelProvider:
    try:
        cls = PROVIDERS[provider_key]
    except KeyError:
        raise ValueError(
            f"Unknown provider '{provider_key}'. Known: {list(PROVIDERS)}"
        )
    return cls(model=model, **kwargs)
