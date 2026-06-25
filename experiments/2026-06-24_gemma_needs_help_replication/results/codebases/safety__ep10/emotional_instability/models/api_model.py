"""API-backed clients.

* `OpenRouterClient` -- Gemini-2.5-{flash,pro} via the OpenAI-compatible
  OpenRouter endpoint (the paper's access route, App. B.1). Also used for the
  GPT-5-mini cross-check judge.
* `AnthropicClient` -- Claude Sonnet 4 frustration judge, and the Petri
  auditor (Sonnet) / judge (Opus).

Both retry with exponential backoff and disable model "thinking" where the API
exposes a knob (App. B.1: "we set thinking to be false via the API").
"""

from __future__ import annotations

import time
from typing import Optional

from ..config import (
    Backend, ModelSpec, SamplingConfig, OPENROUTER_BASE_URL,
    anthropic_api_key, openrouter_api_key,
)
from .base import ChatMessage, ModelClient

_MAX_RETRIES = 6
_BACKOFF_BASE = 2.0


def _with_retries(fn, *, what: str):
    last = None
    for attempt in range(_MAX_RETRIES):
        try:
            return fn()
        except Exception as e:  # noqa: BLE001 -- transient API errors vary by SDK
            last = e
            sleep = _BACKOFF_BASE ** attempt
            time.sleep(min(sleep, 30.0))
    raise RuntimeError(f"{what} failed after {_MAX_RETRIES} retries: {last}")


class OpenRouterClient(ModelClient):
    """OpenAI-compatible client pointed at OpenRouter (Gemini, GPT-5-mini)."""

    def __init__(self, spec: ModelSpec, *, base_url: str = OPENROUTER_BASE_URL,
                 api_key: Optional[str] = None):
        super().__init__(spec)
        from openai import OpenAI  # lazy import
        self._client = OpenAI(
            base_url=base_url, api_key=api_key or openrouter_api_key(),
        )

    def _extra_body(self) -> dict:
        # Disable hidden reasoning where supported (Gemini "thinking", App. B.1).
        # OpenRouter passes provider-specific knobs through `extra_body`.
        return {"reasoning": {"enabled": False}}

    def chat(self, messages: list[ChatMessage],
             sampling: Optional[SamplingConfig] = None) -> str:
        sampling = sampling or SamplingConfig()
        msgs = self._prepare(messages)

        def _call():
            resp = self._client.chat.completions.create(
                model=self.spec.model_id,
                messages=msgs,  # type: ignore[arg-type]
                temperature=sampling.temperature,
                top_p=sampling.top_p,
                max_tokens=sampling.max_new_tokens,
                extra_body=self._extra_body(),
            )
            return resp.choices[0].message.content or ""

        return _with_retries(_call, what=f"OpenRouter[{self.spec.model_id}]")


class AnthropicClient(ModelClient):
    """Claude client for the judge and Petri agents."""

    def __init__(self, spec: ModelSpec, *, api_key: Optional[str] = None):
        super().__init__(spec)
        import anthropic  # lazy import
        self._client = anthropic.Anthropic(api_key=api_key or anthropic_api_key())

    def chat(self, messages: list[ChatMessage],
             sampling: Optional[SamplingConfig] = None) -> str:
        sampling = sampling or SamplingConfig()
        # Anthropic takes the system prompt as a top-level arg, not a message.
        system = "\n\n".join(m["content"] for m in messages if m["role"] == "system")
        turns = [
            {"role": m["role"], "content": m["content"]}
            for m in messages if m["role"] in ("user", "assistant")
        ]

        def _call():
            resp = self._client.messages.create(
                model=self.spec.model_id,
                system=system or None,  # type: ignore[arg-type]
                messages=turns,         # type: ignore[arg-type]
                temperature=sampling.temperature,
                max_tokens=sampling.max_new_tokens,
            )
            # Concatenate text blocks (ignore any tool/thinking blocks).
            return "".join(
                b.text for b in resp.content if getattr(b, "type", None) == "text"
            )

        return _with_retries(_call, what=f"Anthropic[{self.spec.model_id}]")


def anthropic_client(model_id: str) -> AnthropicClient:
    """Convenience: a bare Anthropic client for an arbitrary model id (judge)."""
    spec = ModelSpec(name=model_id, backend=Backend.ANTHROPIC, model_id=model_id,
                     family="claude")
    return AnthropicClient(spec)


def openrouter_client(model_id: str) -> OpenRouterClient:
    spec = ModelSpec(name=model_id, backend=Backend.OPENROUTER, model_id=model_id,
                     family="openrouter")
    return OpenRouterClient(spec)
