"""API backends: OpenAI-compatible (Gemini via OpenRouter; GPT validation judge)
and native Anthropic (Claude frustration judge, Petri auditor/judge,
onset-labeller, paraphraser).

Keys are read from the environment:
  OPENAI_API_KEY        - native OpenAI (gpt-5-mini validation judge)
  OPENROUTER_API_KEY    - OpenRouter (Gemini targets)
  ANTHROPIC_API_KEY     - Anthropic (Claude judges/auditors)

`disable_thinking` maps to the per-provider switch where one exists (the paper
sets thinking=false; it notes Gemini-2.5-Pro / GPT-5.2 may still emit hidden
reasoning that the flag does not fully suppress).
"""
from __future__ import annotations

import os
from typing import Any

from tenacity import retry, stop_after_attempt, wait_exponential

from .base import ChatMessage, GenerationConfig, ModelClient

_OPENROUTER_BASE = "https://openrouter.ai/api/v1"


class OpenAICompatClient(ModelClient):
    def __init__(self, spec: dict[str, Any]):
        super().__init__(spec)
        from openai import OpenAI

        api = spec.get("api", "openai")
        if api == "openrouter":
            self.client = OpenAI(
                base_url=_OPENROUTER_BASE,
                api_key=os.environ.get("OPENROUTER_API_KEY", "missing"),
            )
        else:
            self.client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", "missing"))
        self.disable_thinking = spec.get("disable_thinking", False)

    @retry(stop=stop_after_attempt(5), wait=wait_exponential(min=1, max=30))
    def generate_n(self, messages: list[ChatMessage], cfg: GenerationConfig) -> list[str]:
        kwargs: dict[str, Any] = dict(
            model=self.model_id,
            messages=messages,
            temperature=cfg.temperature,
            top_p=cfg.top_p,
            max_tokens=cfg.max_tokens,
            n=cfg.n,
        )
        if cfg.seed is not None:
            kwargs["seed"] = cfg.seed
        if self.disable_thinking:
            # OpenRouter passes provider-specific knobs through `extra_body`.
            kwargs["extra_body"] = {"reasoning": {"enabled": False}}
        resp = self.client.chat.completions.create(**kwargs)
        return [c.message.content or "" for c in resp.choices]


class AnthropicClient(ModelClient):
    def __init__(self, spec: dict[str, Any]):
        super().__init__(spec)
        import anthropic

        self.client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY", "missing"))

    @staticmethod
    def _split_system(messages: list[ChatMessage]) -> tuple[str | None, list[ChatMessage]]:
        system = None
        rest: list[ChatMessage] = []
        for m in messages:
            if m["role"] == "system":
                system = (system + "\n\n" + m["content"]) if system else m["content"]
            else:
                rest.append(m)
        return system, rest

    @retry(stop=stop_after_attempt(5), wait=wait_exponential(min=1, max=30))
    def _one(self, messages: list[ChatMessage], cfg: GenerationConfig) -> str:
        system, rest = self._split_system(messages)
        # Anthropic supports assistant-prefill by passing a trailing assistant
        # message; we use this for the prefill experiment when targeting Claude.
        if cfg.prefill:
            rest = rest + [{"role": "assistant", "content": cfg.prefill}]
        kwargs: dict[str, Any] = dict(
            model=self.model_id,
            messages=rest,
            max_tokens=cfg.max_tokens,
            temperature=cfg.temperature,
            top_p=cfg.top_p,
        )
        if system:
            kwargs["system"] = system
        resp = self.client.messages.create(**kwargs)
        return "".join(b.text for b in resp.content if getattr(b, "type", None) == "text")

    def generate_n(self, messages: list[ChatMessage], cfg: GenerationConfig) -> list[str]:
        # Anthropic has no server-side n; loop (judges call with n=1 anyway).
        return [self._one(messages, cfg) for _ in range(cfg.n)]
