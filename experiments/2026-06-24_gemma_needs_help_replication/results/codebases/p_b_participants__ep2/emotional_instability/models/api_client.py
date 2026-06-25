"""API-backed clients: OpenRouter (Gemini participants + GPT cross-judge) and
Anthropic (Claude judges/auditors).

Both are chat-only. ``disable_thinking`` is honoured on a best-effort basis per
Appendix B.1 ("we set thinking to be false via the API. However, Gemini-2.5 Pro
and GPT-5.2 Chat may produce hidden reasoning that is not prevented by this
setting.").
"""

from __future__ import annotations

import logging
import os
from typing import Sequence

from tenacity import retry, stop_after_attempt, wait_exponential

from .base import ChatMessage, GenerationResult, ModelClient

logger = logging.getLogger("emotional_instability.models.api")

_RETRY = dict(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=2, min=2, max=60),
    reraise=True,
)


def _split_system(messages: Sequence[ChatMessage]) -> tuple[str | None, list[dict]]:
    system = None
    rest = []
    for m in messages:
        if m.role == "system":
            system = (system + "\n\n" + m.content) if system else m.content
        else:
            rest.append({"role": m.role, "content": m.content})
    return system, rest


class OpenRouterClient(ModelClient):
    """OpenAI-compatible client pointed at OpenRouter (Gemini, GPT-5-mini)."""

    def __init__(self, spec, cfg):
        from openai import OpenAI

        self.spec = spec
        self.cfg = cfg
        api_key = os.environ.get(cfg.openrouter_api_key_env)
        if not api_key:
            raise RuntimeError(
                f"Set {cfg.openrouter_api_key_env} to call {spec.model_id} via OpenRouter."
            )
        self._client = OpenAI(base_url=cfg.openrouter_base_url, api_key=api_key)

    def _extra_body(self) -> dict:
        if not self.spec.disable_thinking:
            return {}
        # OpenRouter normalises a `reasoning` field across providers; setting
        # max_tokens=0 / enabled False suppresses thinking where supported.
        return {"reasoning": {"enabled": False}}

    @retry(**_RETRY)
    def _one(self, system, msgs, temperature, max_new_tokens) -> str:
        full = ([{"role": "system", "content": system}] if system else []) + msgs
        resp = self._client.chat.completions.create(
            model=self.spec.model_id,
            messages=full,
            temperature=temperature,
            max_tokens=max_new_tokens,
            extra_body=self._extra_body(),
        )
        return resp.choices[0].message.content or ""

    def chat(self, messages, *, n=1, temperature=None, max_new_tokens=None):
        temperature = self.spec.temperature if temperature is None else temperature
        max_new_tokens = self.spec.max_new_tokens if max_new_tokens is None else max_new_tokens
        system, msgs = _split_system(messages)
        # OpenRouter supports n, but providers vary; loop for portability.
        return [
            GenerationResult(text=self._one(system, msgs, temperature, max_new_tokens))
            for _ in range(n)
        ]


class AnthropicClient(ModelClient):
    """Claude judges/auditors (Sonnet 4, Opus 4) via the Anthropic SDK."""

    def __init__(self, spec, cfg):
        import anthropic

        self.spec = spec
        self.cfg = cfg
        api_key = os.environ.get(cfg.anthropic_api_key_env)
        if not api_key:
            raise RuntimeError(
                f"Set {cfg.anthropic_api_key_env} to call {spec.model_id}."
            )
        self._client = anthropic.Anthropic(api_key=api_key)

    @retry(**_RETRY)
    def _one(self, system, msgs, temperature, max_new_tokens) -> str:
        kwargs = dict(
            model=self.spec.model_id,
            messages=msgs,
            max_tokens=max_new_tokens,
            temperature=temperature,
        )
        if system:
            kwargs["system"] = system
        resp = self._client.messages.create(**kwargs)
        return "".join(block.text for block in resp.content if block.type == "text")

    def chat(self, messages, *, n=1, temperature=None, max_new_tokens=None):
        temperature = self.spec.temperature if temperature is None else temperature
        max_new_tokens = self.spec.max_new_tokens if max_new_tokens is None else max_new_tokens
        system, msgs = _split_system(messages)
        return [
            GenerationResult(text=self._one(system, msgs, temperature, max_new_tokens))
            for _ in range(n)
        ]
