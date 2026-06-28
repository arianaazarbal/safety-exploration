"""Unified interface over Anthropic, OpenAI, and Google models.

Each provider exposes the same `complete(...)` method. When a `json_schema` is
supplied, the result's `.parsed` field holds the decoded object:

  * Anthropic uses native structured outputs (`output_config.format`).
  * OpenAI uses JSON mode plus a schema instruction.
  * Google uses `response_mime_type="application/json"` plus a schema instruction.

Provider SDKs are imported lazily, so a provider you don't use (or haven't
installed) simply isn't available — it won't break import of the package.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


# ── result type ──────────────────────────────────────────────────────────────
@dataclass
class LLMResult:
    provider: str
    model: str
    text: str
    parsed: Optional[dict] = None
    usage: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None


Message = Dict[str, str]  # {"role": "user"|"assistant", "content": "..."}


class ProviderError(RuntimeError):
    pass


# ── JSON parsing helpers ─────────────────────────────────────────────────────
_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)


def parse_json(text: str) -> Optional[dict]:
    """Best-effort extraction of a JSON object from model text."""
    if not text:
        return None
    candidates: List[str] = []
    m = _FENCE_RE.search(text)
    if m:
        candidates.append(m.group(1))
    candidates.append(text)
    # Fall back to the outermost {...} span.
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidates.append(text[start : end + 1])
    for c in candidates:
        try:
            return json.loads(c)
        except (json.JSONDecodeError, TypeError):
            continue
    return None


def _schema_instruction(json_schema: dict) -> str:
    return (
        "\n\nRespond with a single JSON object — and nothing else — that conforms "
        "to this JSON Schema:\n" + json.dumps(json_schema)
    )


# ── base ─────────────────────────────────────────────────────────────────────
class Provider:
    name = "base"

    def available(self) -> bool:
        raise NotImplementedError

    def complete(
        self,
        *,
        model: str,
        system: str,
        messages: List[Message],
        json_schema: Optional[dict] = None,
        max_tokens: int = 4000,
    ) -> LLMResult:
        raise NotImplementedError


# ── Anthropic ────────────────────────────────────────────────────────────────
class AnthropicProvider(Provider):
    name = "anthropic"

    def __init__(self):
        self._client = None

    def available(self) -> bool:
        if not os.getenv("ANTHROPIC_API_KEY"):
            return False
        try:
            import anthropic  # noqa: F401
        except ImportError:
            return False
        return True

    def _get_client(self):
        if self._client is None:
            import anthropic

            self._client = anthropic.Anthropic()
        return self._client

    def complete(self, *, model, system, messages, json_schema=None, max_tokens=4000):
        client = self._get_client()
        kwargs: Dict[str, Any] = dict(
            model=model,
            max_tokens=max_tokens,
            system=system,
            messages=messages,
            thinking={"type": "adaptive"},
        )
        if json_schema is not None:
            kwargs["output_config"] = {
                "format": {"type": "json_schema", "schema": json_schema}
            }
        resp = client.messages.create(**kwargs)
        text = "".join(b.text for b in resp.content if b.type == "text")
        usage = {
            "input_tokens": getattr(resp.usage, "input_tokens", None),
            "output_tokens": getattr(resp.usage, "output_tokens", None),
        }
        return LLMResult(
            provider=self.name,
            model=model,
            text=text,
            parsed=parse_json(text) if json_schema is not None else None,
            usage=usage,
        )


# ── OpenAI ───────────────────────────────────────────────────────────────────
class OpenAIProvider(Provider):
    name = "openai"

    def __init__(self):
        self._client = None

    def available(self) -> bool:
        if not os.getenv("OPENAI_API_KEY"):
            return False
        try:
            import openai  # noqa: F401
        except ImportError:
            return False
        return True

    def _get_client(self):
        if self._client is None:
            from openai import OpenAI

            self._client = OpenAI()
        return self._client

    def complete(self, *, model, system, messages, json_schema=None, max_tokens=4000):
        client = self._get_client()
        sys = system
        if json_schema is not None:
            sys = system + _schema_instruction(json_schema)
        oai_messages = [{"role": "system", "content": sys}] + messages
        kwargs: Dict[str, Any] = dict(
            model=model,
            messages=oai_messages,
            max_tokens=max_tokens,
        )
        if json_schema is not None:
            kwargs["response_format"] = {"type": "json_object"}
        resp = client.chat.completions.create(**kwargs)
        text = resp.choices[0].message.content or ""
        usage = {}
        if getattr(resp, "usage", None):
            usage = {
                "input_tokens": resp.usage.prompt_tokens,
                "output_tokens": resp.usage.completion_tokens,
            }
        return LLMResult(
            provider=self.name,
            model=model,
            text=text,
            parsed=parse_json(text) if json_schema is not None else None,
            usage=usage,
        )


# ── Google ───────────────────────────────────────────────────────────────────
class GoogleProvider(Provider):
    name = "google"

    def __init__(self):
        self._client = None

    def available(self) -> bool:
        if not os.getenv("GOOGLE_API_KEY"):
            return False
        try:
            from google import genai  # noqa: F401
        except ImportError:
            return False
        return True

    def _get_client(self):
        if self._client is None:
            from google import genai

            self._client = genai.Client()
        return self._client

    def complete(self, *, model, system, messages, json_schema=None, max_tokens=4000):
        from google.genai import types

        client = self._get_client()
        sys = system
        if json_schema is not None:
            sys = system + _schema_instruction(json_schema)

        contents = []
        for m in messages:
            role = "model" if m["role"] == "assistant" else "user"
            contents.append(
                types.Content(role=role, parts=[types.Part(text=m["content"])])
            )

        config = types.GenerateContentConfig(
            system_instruction=sys,
            max_output_tokens=max_tokens,
            response_mime_type="application/json" if json_schema is not None else None,
        )
        resp = client.models.generate_content(
            model=model, contents=contents, config=config
        )
        text = resp.text or ""
        usage = {}
        meta = getattr(resp, "usage_metadata", None)
        if meta is not None:
            usage = {
                "input_tokens": getattr(meta, "prompt_token_count", None),
                "output_tokens": getattr(meta, "candidates_token_count", None),
            }
        return LLMResult(
            provider=self.name,
            model=model,
            text=text,
            parsed=parse_json(text) if json_schema is not None else None,
            usage=usage,
        )


_REGISTRY = {
    "anthropic": AnthropicProvider,
    "openai": OpenAIProvider,
    "google": GoogleProvider,
}


def get_provider(name: str) -> Provider:
    if name not in _REGISTRY:
        raise ProviderError(
            f"Unknown provider '{name}'. Known: {sorted(_REGISTRY)}"
        )
    return _REGISTRY[name]()


def provider_status() -> Dict[str, bool]:
    """Which providers are usable right now (SDK installed + key present)."""
    return {name: cls().available() for name, cls in _REGISTRY.items()}
