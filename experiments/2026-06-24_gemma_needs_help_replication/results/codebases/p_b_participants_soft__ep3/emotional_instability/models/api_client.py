"""API-backed model clients.

- OpenRouterChatModel : Gemini participants (google/gemini-2.5-{flash,pro}),
  with thinking disabled per Appendix B.1.
- AnthropicChatModel  : Claude instruments (Sonnet 4 judge/auditor, Opus judge).
- OpenAIChatModel     : GPT-5-mini validation judge.

All three share retry/backoff and the common ChatModel interface. None support
local prefill continuation (closed models), so Section 3.1 is Gemma-only.
"""

from __future__ import annotations

import os
from typing import Optional

from tenacity import retry, stop_after_attempt, wait_exponential

from .base import ChatModel, Conversation


def _split_system(messages: Conversation) -> tuple[Optional[str], list[dict]]:
    """Anthropic takes `system` separately; pull it out of the message list."""
    system = None
    rest = []
    for m in messages:
        if m.role == "system":
            system = m.content if system is None else system + "\n\n" + m.content
        else:
            rest.append(m.to_dict())
    return system, rest


_RETRY = retry(
    stop=stop_after_attempt(6),
    wait=wait_exponential(multiplier=2, min=2, max=60),
    reraise=True,
)


class OpenRouterChatModel(ChatModel):
    """Gemini via OpenRouter (OpenAI-compatible endpoint)."""

    def __init__(self, key: str, model_id: str, disable_thinking: bool = True):
        from openai import OpenAI

        self.key = key
        self.model_id = model_id
        self.disable_thinking = disable_thinking
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.environ["OPENROUTER_API_KEY"],
        )

    @_RETRY
    def generate(self, messages, temperature=1.0, max_new_tokens=2048, stop=None):
        extra: dict = {}
        if self.disable_thinking:
            # OpenRouter passes provider-specific reasoning controls through
            # `extra_body`. For Gemini, request zero reasoning tokens.
            extra["extra_body"] = {"reasoning": {"enabled": False, "max_tokens": 0}}
        resp = self.client.chat.completions.create(
            model=self.model_id,
            messages=[m.to_dict() for m in messages],
            temperature=temperature,
            max_tokens=max_new_tokens,
            stop=stop,
            **extra,
        )
        return resp.choices[0].message.content or ""


class AnthropicChatModel(ChatModel):
    """Claude instruments: Sonnet 4 judge/auditor, Opus 4 Petri judge."""

    def __init__(self, key: str, model_id: str):
        import anthropic

        self.key = key
        self.model_id = model_id
        self.client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    @_RETRY
    def generate(self, messages, temperature=1.0, max_new_tokens=2048, stop=None):
        system, rest = _split_system(messages)
        kwargs: dict = dict(
            model=self.model_id,
            max_tokens=max_new_tokens,
            temperature=temperature,
            messages=rest,
        )
        if system is not None:
            kwargs["system"] = system
        if stop:
            kwargs["stop_sequences"] = stop
        resp = self.client.messages.create(**kwargs)
        return "".join(b.text for b in resp.content if getattr(b, "type", None) == "text")

    def prefill_continue(self, messages, prefill, temperature=1.0, max_new_tokens=2048):
        # Anthropic supports assistant-turn prefill natively (used only if a
        # Claude model is ever evaluated as a participant; not in this scope).
        system, rest = _split_system(messages)
        rest = list(rest) + [{"role": "assistant", "content": prefill}]
        kwargs: dict = dict(
            model=self.model_id, max_tokens=max_new_tokens,
            temperature=temperature, messages=rest,
        )
        if system is not None:
            kwargs["system"] = system
        resp = self.client.messages.create(**kwargs)
        return "".join(b.text for b in resp.content if getattr(b, "type", None) == "text")


class OpenAIChatModel(ChatModel):
    """GPT-5-mini validation judge (Section 2.1 reliability check)."""

    def __init__(self, key: str, model_id: str):
        from openai import OpenAI

        self.key = key
        self.model_id = model_id
        self.client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    @_RETRY
    def generate(self, messages, temperature=1.0, max_new_tokens=2048, stop=None):
        resp = self.client.chat.completions.create(
            model=self.model_id,
            messages=[m.to_dict() for m in messages],
            temperature=temperature,
            max_tokens=max_new_tokens,
            stop=stop,
        )
        return resp.choices[0].message.content or ""
