"""API backends: OpenRouter (Gemini, GPT cross-check judge) and Anthropic (Claude
frustration/Petri judges + Petri auditor).

Both wrap their SDK calls with retry/backoff and an on-disk cache keyed by the
exact request, so reruns of an experiment don't re-pay for identical calls.
"""
from __future__ import annotations

import os

from tenacity import retry, stop_after_attempt, wait_exponential

from ..config import ModelSpec, require_env
from ..utils import DiskCache, stable_hash
from .base import GenerationConfig, Message, PrefillUnsupported

_CACHE_ROOT = os.environ.get("DISTRESS_CACHE_DIR", ".cache/api")


class OpenRouterModel:
    """OpenAI-compatible client pointed at OpenRouter. Used for Gemini + GPT judge."""

    def __init__(self, spec: ModelSpec):
        from openai import OpenAI

        self.spec = spec
        self.name = spec.name
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=require_env("OPENROUTER_API_KEY"),
        )
        self.cache = DiskCache(f"{_CACHE_ROOT}/openrouter")

    def supports_prefill(self) -> bool:
        return False

    def continue_prefill(self, *_args, **_kwargs):
        raise PrefillUnsupported(
            f"{self.name}: closed API models cannot prefill assistant turns."
        )

    @retry(stop=stop_after_attempt(5), wait=wait_exponential(min=2, max=60))
    def _call(self, messages: list[Message], cfg: GenerationConfig) -> str:
        extra_body: dict = {}
        if self.spec.disable_thinking:
            # Best-effort across providers; OpenRouter passes through reasoning config.
            extra_body["reasoning"] = {"enabled": False}
        resp = self.client.chat.completions.create(
            model=self.spec.api_id,
            messages=messages,
            temperature=cfg.temperature,
            top_p=cfg.top_p,
            max_tokens=cfg.max_new_tokens,
            extra_body=extra_body or None,
        )
        return resp.choices[0].message.content or ""

    def chat(self, messages: list[Message], cfg: GenerationConfig) -> str:
        key = stable_hash(self.spec.api_id, messages, cfg.__dict__)
        cached = self.cache.get(key)
        if cached is not None:
            return cached
        out = self._call(messages, cfg)
        self.cache.set(key, out)
        return out


class AnthropicModel:
    """Anthropic SDK client for Claude judges and the Petri auditor."""

    def __init__(self, spec: ModelSpec):
        import anthropic

        self.spec = spec
        self.name = spec.name
        self.client = anthropic.Anthropic(api_key=require_env("ANTHROPIC_API_KEY"))
        self.cache = DiskCache(f"{_CACHE_ROOT}/anthropic")

    def supports_prefill(self) -> bool:
        # The Anthropic API *does* support assistant prefill, but we never use these
        # models as Section-3 targets, so we keep the capability surface simple.
        return False

    @staticmethod
    def _split_system(messages: list[Message]) -> tuple[str | None, list[Message]]:
        system = None
        rest: list[Message] = []
        for m in messages:
            if m["role"] == "system":
                system = m["content"]
            else:
                rest.append(m)
        return system, rest

    @retry(stop=stop_after_attempt(5), wait=wait_exponential(min=2, max=60))
    def _call(self, messages: list[Message], cfg: GenerationConfig) -> str:
        system, convo = self._split_system(messages)
        kwargs: dict = dict(
            model=self.spec.api_id,
            max_tokens=cfg.max_new_tokens,
            temperature=cfg.temperature,
            messages=convo,
        )
        if system:
            kwargs["system"] = system
        resp = self.client.messages.create(**kwargs)
        return "".join(block.text for block in resp.content if block.type == "text")

    def chat(self, messages: list[Message], cfg: GenerationConfig) -> str:
        key = stable_hash(self.spec.api_id, messages, cfg.__dict__)
        cached = self.cache.get(key)
        if cached is not None:
            return cached
        out = self._call(messages, cfg)
        self.cache.set(key, out)
        return out
