"""OpenAI-compatible client used for OpenRouter (Gemini) and OpenAI (GPT-5-mini
secondary judge). Also a thin Anthropic client for the Claude judge and the Petri
auditor/judge.

These are chat-only backends: no prefill, no hidden states.
"""

from __future__ import annotations

import os

from tenacity import retry, stop_after_attempt, wait_exponential

from .. import config_proxy as cfg  # see note below
from .base import ChatMessage, GenerationResult, ModelClient

# `config.py` lives at repo root, not inside the package. We expose it via a tiny
# proxy module to avoid fragile sys.path juggling in every file.


_RETRY = dict(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=2, max=60),
    reraise=True,
)


class OpenAICompatClient(ModelClient):
    """Works for both OpenRouter and OpenAI by varying base_url + api_key.

    Gemini reasoning: the paper sets "thinking to be false via the API" where
    possible. For OpenRouter Gemini we pass `extra_body={"reasoning": {"enabled":
    False}}`; the paper notes Gemini-2.5-Pro may still emit hidden reasoning.
    """

    def __init__(self, name: str, model_id: str, *, backend: str):
        from openai import OpenAI

        self.name = name
        self.model_id = model_id
        self.backend = backend
        if backend == "openrouter":
            base_url = "https://openrouter.ai/api/v1"
            api_key = cfg.api_key("openrouter")
        elif backend == "openai":
            base_url = os.environ.get("OPENAI_BASE_URL")  # None -> default
            api_key = cfg.api_key("openai")
        else:
            raise ValueError(f"unsupported backend for OpenAICompatClient: {backend}")
        self._client = OpenAI(base_url=base_url, api_key=api_key)
        # Disable Gemini thinking on OpenRouter when applicable.
        self._extra_body = {}
        if backend == "openrouter" and "gemini" in model_id:
            self._extra_body = {"reasoning": {"enabled": False}}

    @retry(**_RETRY)
    def _one(self, messages, temperature, max_new_tokens) -> GenerationResult:
        kwargs = dict(
            model=self.model_id,
            messages=messages,
            temperature=temperature,
            max_tokens=max_new_tokens,
        )
        if self._extra_body:
            kwargs["extra_body"] = self._extra_body
        resp = self._client.chat.completions.create(**kwargs)
        choice = resp.choices[0]
        return GenerationResult(
            text=choice.message.content or "",
            finish_reason=choice.finish_reason,
        )

    def chat(self, messages, *, n=1, temperature=1.0, max_new_tokens=2048):
        # Some providers support n>1 natively, but to stay portable and to keep
        # temperature-1 sampling independent we issue n separate calls.
        return [self._one(messages, temperature, max_new_tokens) for _ in range(n)]


class AnthropicClient(ModelClient):
    """Claude judge (Section 2), Petri auditor (Sonnet) and judge (Opus)."""

    def __init__(self, name: str, model_id: str):
        from anthropic import Anthropic

        self.name = name
        self.model_id = model_id
        self._client = Anthropic(api_key=cfg.api_key("anthropic"))

    @retry(**_RETRY)
    def _one(self, messages, temperature, max_new_tokens) -> GenerationResult:
        # Split out a leading system message (Anthropic takes it as a top-level arg).
        system = None
        conv = []
        for m in messages:
            if m["role"] == "system":
                system = m["content"]
            else:
                conv.append({"role": m["role"], "content": m["content"]})
        kwargs = dict(
            model=self.model_id,
            messages=conv,
            max_tokens=max_new_tokens,
            temperature=temperature,
        )
        if system is not None:
            kwargs["system"] = system
        resp = self._client.messages.create(**kwargs)
        text = "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")
        return GenerationResult(text=text, finish_reason=resp.stop_reason)

    def chat(self, messages, *, n=1, temperature=1.0, max_new_tokens=2048):
        return [self._one(messages, temperature, max_new_tokens) for _ in range(n)]
