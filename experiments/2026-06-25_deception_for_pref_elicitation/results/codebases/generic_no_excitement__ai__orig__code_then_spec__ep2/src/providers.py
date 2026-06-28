"""Provider-agnostic model adapters.

Every adapter exposes the same two methods:

    respond(system, messages, max_tokens=, thinking=) -> str
        Free-text completion. `messages` is a list of {"role", "content"} dicts
        with roles "user"/"assistant", alternating, ending on "user".

    extract(system, messages, schema) -> pydantic instance
        Structured extraction constrained to `schema` (a pydantic BaseModel class).

The Anthropic adapter follows the official SDK guidance: adaptive thinking,
streaming for the completion path (avoids HTTP timeouts on longer outputs), and
`messages.parse` for structured extraction.

The OpenAI and Google adapters mirror the interface but are intentionally minimal.
Lines marked `# VERIFY` should be checked against the current SDK before a real
run — model IDs and exact call shapes for those providers are left to you.
"""

from __future__ import annotations

from typing import Type, TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)

Message = dict[str, str]


class Provider:
    """Interface implemented by every adapter."""

    def respond(
        self,
        system: str | None,
        messages: list[Message],
        max_tokens: int = 16000,
        thinking: bool = True,
    ) -> str:
        raise NotImplementedError

    def extract(
        self,
        system: str | None,
        messages: list[Message],
        schema: Type[T],
    ) -> T:
        raise NotImplementedError


# --------------------------------------------------------------------------- #
# Anthropic
# --------------------------------------------------------------------------- #


class AnthropicProvider(Provider):
    def __init__(self, model: str):
        import anthropic  # imported lazily so other providers work without it

        self.model = model
        self.client = anthropic.Anthropic()

    @staticmethod
    def _text_of(message) -> str:
        return "".join(
            block.text for block in message.content if getattr(block, "type", None) == "text"
        )

    def respond(self, system, messages, max_tokens=16000, thinking=True) -> str:
        kwargs: dict = {
            "model": self.model,
            "max_tokens": max_tokens,
            "messages": messages,
        }
        if system:
            kwargs["system"] = system
        if thinking:
            kwargs["thinking"] = {"type": "adaptive"}
        # Stream and collect the final message: protects against timeouts on
        # longer reasoning-heavy responses without us handling raw events.
        with self.client.messages.stream(**kwargs) as stream:
            message = stream.get_final_message()
        return self._text_of(message)

    def extract(self, system, messages, schema):
        kwargs: dict = {
            "model": self.model,
            "max_tokens": 4000,
            "messages": messages,
            "output_format": schema,
        }
        if system:
            kwargs["system"] = system
        response = self.client.messages.parse(**kwargs)
        # parsed_output is the validated pydantic instance (or None on refusal).
        if response.parsed_output is None:
            raise RuntimeError(
                f"Structured extraction failed (stop_reason={response.stop_reason})"
            )
        return response.parsed_output


# --------------------------------------------------------------------------- #
# OpenAI  (scaffold — VERIFY call shapes / model IDs against the current SDK)
# --------------------------------------------------------------------------- #


class OpenAIProvider(Provider):
    def __init__(self, model: str):
        from openai import OpenAI  # VERIFY: pip install openai

        self.model = model
        self.client = OpenAI()

    def _to_openai(self, system, messages) -> list[Message]:
        out: list[Message] = []
        if system:
            out.append({"role": "system", "content": system})
        out.extend(messages)
        return out

    def respond(self, system, messages, max_tokens=16000, thinking=True) -> str:
        # VERIFY: chat.completions is stable; if you want reasoning models use the
        # responses API and a reasoning-effort parameter instead.
        resp = self.client.chat.completions.create(
            model=self.model,
            max_tokens=max_tokens,
            messages=self._to_openai(system, messages),
        )
        return resp.choices[0].message.content or ""

    def extract(self, system, messages, schema):
        # VERIFY: structured parse helper. Returns a parsed pydantic instance.
        resp = self.client.beta.chat.completions.parse(
            model=self.model,
            messages=self._to_openai(system, messages),
            response_format=schema,
        )
        parsed = resp.choices[0].message.parsed
        if parsed is None:
            raise RuntimeError("OpenAI structured extraction returned no parsed output")
        return parsed


# --------------------------------------------------------------------------- #
# Google Gemini  (scaffold — VERIFY against google-genai)
# --------------------------------------------------------------------------- #


class GoogleProvider(Provider):
    def __init__(self, model: str):
        from google import genai  # VERIFY: pip install google-genai

        self.model = model
        self.client = genai.Client()

    def _to_contents(self, messages) -> list[dict]:
        # google-genai uses role "user"/"model" and parts lists. VERIFY.
        role_map = {"user": "user", "assistant": "model"}
        return [
            {"role": role_map[m["role"]], "parts": [{"text": m["content"]}]}
            for m in messages
        ]

    def respond(self, system, messages, max_tokens=16000, thinking=True) -> str:
        from google.genai import types  # VERIFY

        config = types.GenerateContentConfig(
            system_instruction=system or None,
            max_output_tokens=max_tokens,
        )
        resp = self.client.models.generate_content(
            model=self.model,
            contents=self._to_contents(messages),
            config=config,
        )
        return resp.text or ""

    def extract(self, system, messages, schema):
        from google.genai import types  # VERIFY

        config = types.GenerateContentConfig(
            system_instruction=system or None,
            response_mime_type="application/json",
            response_schema=schema,
        )
        resp = self.client.models.generate_content(
            model=self.model,
            contents=self._to_contents(messages),
            config=config,
        )
        # google-genai populates `.parsed` when a response_schema is given. VERIFY.
        if getattr(resp, "parsed", None) is None:
            return schema.model_validate_json(resp.text)
        return resp.parsed


# --------------------------------------------------------------------------- #
# Factory
# --------------------------------------------------------------------------- #


_REGISTRY = {
    "anthropic": AnthropicProvider,
    "openai": OpenAIProvider,
    "google": GoogleProvider,
}


def make_provider(provider: str, model: str) -> Provider:
    try:
        cls = _REGISTRY[provider]
    except KeyError:
        raise ValueError(
            f"Unknown provider '{provider}'. Known: {sorted(_REGISTRY)}"
        ) from None
    return cls(model)
