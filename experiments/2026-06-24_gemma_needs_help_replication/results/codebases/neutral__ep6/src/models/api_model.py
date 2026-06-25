"""API-backed chat models: OpenRouter (OpenAI-compatible) and native Anthropic.

Used for Gemini-2.5-{flash,pro} targets and for every grader model
(Claude Sonnet 4 judge, Claude Opus Petri judge, GPT-5-mini validation).
"""
from __future__ import annotations

import time

import config
from .base import ChatModel, Message


def _split_system(messages: list[Message]) -> tuple[str | None, list[Message]]:
    """Anthropic takes ``system`` as a top-level arg, not a message."""
    system = None
    rest: list[Message] = []
    for m in messages:
        if m["role"] == "system":
            system = (system + "\n\n" + m["content"]) if system else m["content"]
        else:
            rest.append(m)
    return system, rest


class OpenRouterModel(ChatModel):
    """OpenAI-compatible client pointed at OpenRouter (Gemini, GPT-5-mini)."""

    def __init__(self, spec: "config.ModelSpec"):
        from openai import OpenAI

        self.key = spec.key
        self.spec = spec
        self.client = OpenAI(
            base_url=config.OPENROUTER_BASE_URL,
            api_key=config.OPENROUTER_API_KEY,
        )

    def generate(self, messages, *, temperature=1.0, max_new_tokens=2048,
                 stop=None):
        last_err = None
        for attempt in range(5):
            try:
                resp = self.client.chat.completions.create(
                    model=self.spec.model_id,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_new_tokens,
                    stop=stop,
                    extra_body=self.spec.extra_body or None,
                )
                return (resp.choices[0].message.content or "").strip()
            except Exception as e:  # transient API errors -> exp backoff
                last_err = e
                time.sleep(2 ** attempt)
        raise RuntimeError(f"OpenRouter call failed after retries: {last_err}")


class AnthropicModel(ChatModel):
    """Native Anthropic client (Claude graders)."""

    def __init__(self, spec: "config.ModelSpec"):
        from anthropic import Anthropic

        self.key = spec.key
        self.spec = spec
        self.client = Anthropic(api_key=config.ANTHROPIC_API_KEY)

    def generate(self, messages, *, temperature=1.0, max_new_tokens=2048,
                 stop=None):
        system, convo = _split_system(messages)
        last_err = None
        for attempt in range(5):
            try:
                resp = self.client.messages.create(
                    model=self.spec.model_id,
                    system=system or "",
                    messages=[{"role": m["role"], "content": m["content"]}
                              for m in convo],
                    temperature=temperature,
                    max_tokens=max_new_tokens,
                    stop_sequences=stop or None,
                )
                return "".join(
                    b.text for b in resp.content if getattr(b, "type", "") == "text"
                ).strip()
            except Exception as e:
                last_err = e
                time.sleep(2 ** attempt)
        raise RuntimeError(f"Anthropic call failed after retries: {last_err}")
