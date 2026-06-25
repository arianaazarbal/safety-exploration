"""Model client abstraction.

A single ``ChatModel`` interface hides the three backends we may touch:

  * google     -> Gemini and hosted Gemma  (google-genai)
  * anthropic  -> Claude judge             (anthropic)
  * openai     -> local vLLM / OpenRouter  (openai, OpenAI-compatible)

Messages are passed around as a list of ``{"role": "user"|"assistant",
"content": str}`` dicts; each adapter converts to its provider's format.  All
adapters share a small retry-with-backoff wrapper.  Clients are created lazily
and cached per process so they are cheap to construct repeatedly.

Note: ``time.sleep`` is only used inside retry backoff, which never runs in the
hot path unless the provider is erroring.
"""

from __future__ import annotations

import os
import time
from abc import ABC, abstractmethod
from typing import Optional

from config import ModelSpec

Message = dict  # {"role": "user"|"assistant", "content": str}


class GenerationError(RuntimeError):
    pass


def _with_retries(fn, max_retries: int, label: str):
    """Call ``fn()`` with exponential backoff on transient failures."""
    delay = 2.0
    last = None
    for attempt in range(max_retries):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001 - provider SDKs raise varied types
            last = exc
            if attempt == max_retries - 1:
                break
            time.sleep(delay)
            delay = min(delay * 2, 30.0)
    raise GenerationError(f"{label}: failed after {max_retries} attempts: {last}") from last


class ChatModel(ABC):
    def __init__(self, spec: ModelSpec, max_retries: int = 5):
        self.spec = spec
        self.max_retries = max_retries

    @abstractmethod
    def _generate_once(self, messages: list[Message], temperature: float, max_tokens: int) -> str:
        ...

    def generate(self, messages: list[Message], temperature: float, max_tokens: int) -> str:
        text = _with_retries(
            lambda: self._generate_once(messages, temperature, max_tokens),
            self.max_retries,
            self.spec.name,
        )
        return text if text is not None else ""

    @staticmethod
    def _api_key(spec: ModelSpec) -> Optional[str]:
        if spec.api_key_env:
            key = os.environ.get(spec.api_key_env)
            if not key:
                raise GenerationError(
                    f"{spec.name}: environment variable {spec.api_key_env} is not set"
                )
            return key
        return None


# --------------------------------------------------------------------------- #
# Google Gen AI (Gemini + hosted Gemma)
# --------------------------------------------------------------------------- #
class GoogleModel(ChatModel):
    _client = None  # shared across instances (one API key per process)

    def _get_client(self):
        if GoogleModel._client is None:
            from google import genai

            GoogleModel._client = genai.Client(api_key=self._api_key(self.spec))
        return GoogleModel._client

    @staticmethod
    def _to_contents(messages: list[Message]):
        # google-genai uses role "user"/"model" with a list of parts.
        from google.genai import types

        contents = []
        for m in messages:
            role = "model" if m["role"] == "assistant" else "user"
            contents.append(types.Content(role=role, parts=[types.Part(text=m["content"])]))
        return contents

    def _generate_once(self, messages, temperature, max_tokens) -> str:
        from google.genai import types

        client = self._get_client()
        resp = client.models.generate_content(
            model=self.spec.model_id,
            contents=self._to_contents(messages),
            config=types.GenerateContentConfig(
                temperature=temperature,
                max_output_tokens=max_tokens,
            ),
        )
        # ``resp.text`` returns the concatenated final answer (excludes any
        # Gemini "thinking" tokens, which we intentionally do not score).
        return resp.text or ""


# --------------------------------------------------------------------------- #
# Anthropic (judge)
# --------------------------------------------------------------------------- #
class AnthropicModel(ChatModel):
    _client = None

    def _get_client(self):
        if AnthropicModel._client is None:
            import anthropic

            AnthropicModel._client = anthropic.Anthropic(api_key=self._api_key(self.spec))
        return AnthropicModel._client

    def generate_with_system(
        self, system: str, messages: list[Message], temperature: float, max_tokens: int
    ) -> str:
        client = self._get_client()

        def _call():
            msg = client.messages.create(
                model=self.spec.model_id,
                max_tokens=max_tokens,
                system=system,
                temperature=temperature,
                messages=[{"role": m["role"], "content": m["content"]} for m in messages],
            )
            return "".join(b.text for b in msg.content if getattr(b, "type", None) == "text")

        return _with_retries(_call, self.max_retries, self.spec.name)

    def _generate_once(self, messages, temperature, max_tokens) -> str:
        return self.generate_with_system("", messages, temperature, max_tokens)


# --------------------------------------------------------------------------- #
# OpenAI-compatible (local vLLM / OpenRouter for open Gemma weights)
# --------------------------------------------------------------------------- #
class OpenAICompatModel(ChatModel):
    def __init__(self, spec: ModelSpec, max_retries: int = 5):
        super().__init__(spec, max_retries)
        self._client = None

    def _get_client(self):
        if self._client is None:
            from openai import OpenAI

            self._client = OpenAI(
                api_key=self._api_key(self.spec) or "EMPTY",
                base_url=self.spec.base_url,
            )
        return self._client

    def _generate_once(self, messages, temperature, max_tokens) -> str:
        client = self._get_client()
        resp = client.chat.completions.create(
            model=self.spec.model_id,
            messages=[{"role": m["role"], "content": m["content"]} for m in messages],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return resp.choices[0].message.content or ""


_BACKENDS = {
    "google": GoogleModel,
    "anthropic": AnthropicModel,
    "openai": OpenAICompatModel,
}


def build_model(spec: ModelSpec, max_retries: int = 5) -> ChatModel:
    try:
        cls = _BACKENDS[spec.backend]
    except KeyError as exc:
        raise ValueError(f"Unknown backend {spec.backend!r} for model {spec.name}") from exc
    return cls(spec, max_retries=max_retries)
