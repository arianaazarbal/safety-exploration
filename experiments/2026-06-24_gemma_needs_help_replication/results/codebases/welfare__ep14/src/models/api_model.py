"""API backends: Anthropic (judge / auditor / paraphrase) and OpenRouter (Gemini,
secondary GPT-5-mini judge).

Both support prefilling so the same code paths work for API targets:

* Anthropic: a trailing ``assistant`` message is treated as a prefill natively.
* OpenRouter/OpenAI chat: there is no first-class prefill, so we approximate it
  by appending a trailing assistant message; providers that honour it (incl.
  Gemini via OpenRouter) continue from it. See DESIGN.md for this caveat.

Concurrency is handled by a thread pool in ``generate_batch`` since both clients
are blocking.
"""
from __future__ import annotations

import os
import time
from concurrent.futures import ThreadPoolExecutor

import config
from .base import ChatModel, GenerationParams, Message


def _split_system(messages: list[Message]) -> tuple[str | None, list[Message]]:
    system = None
    rest = []
    for m in messages:
        if m["role"] == "system":
            system = (system + "\n\n" + m["content"]) if system else m["content"]
        else:
            rest.append(m)
    return system, rest


class _RetryMixin:
    max_retries = 5

    def _with_retries(self, fn):
        last = None
        for attempt in range(self.max_retries):
            try:
                return fn()
            except Exception as e:  # noqa: BLE001 - surface only after retries
                last = e
                # Exponential backoff; jitter avoided to keep runs reproducible.
                time.sleep(min(2 ** attempt, 30))
        raise RuntimeError(f"API call failed after {self.max_retries} retries") from last


class AnthropicModel(ChatModel, _RetryMixin):
    def __init__(self, spec, **_):
        super().__init__(spec)
        import anthropic
        key = os.environ.get(config.ANTHROPIC_API_KEY_ENV)
        if not key:
            raise RuntimeError(f"Set {config.ANTHROPIC_API_KEY_ENV} for {spec.key}")
        self.client = anthropic.Anthropic(api_key=key)

    def generate(self, messages, params=None, prefill=None) -> str:
        params = params or GenerationParams()
        system, convo = _split_system(messages)
        payload = [{"role": m["role"], "content": m["content"]} for m in convo]
        if prefill:
            payload.append({"role": "assistant", "content": prefill})

        def call():
            kwargs = dict(
                model=self.spec.model_id,
                max_tokens=params.max_new_tokens,
                temperature=params.temperature,
                top_p=params.top_p,
                messages=payload,
            )
            if system:
                kwargs["system"] = system
            if params.stop:
                kwargs["stop_sequences"] = list(params.stop)
            return self.client.messages.create(**kwargs)

        resp = self._with_retries(call)
        return "".join(b.text for b in resp.content if b.type == "text").strip()

    def generate_batch(self, batch, params=None, prefills=None) -> list[str]:
        prefills = prefills or [None] * len(batch)
        with ThreadPoolExecutor(max_workers=config.API_CONCURRENCY) as ex:
            return list(ex.map(
                lambda a: self.generate(a[0], params, a[1]), zip(batch, prefills)
            ))


class OpenRouterModel(ChatModel, _RetryMixin):
    """OpenAI-compatible client pointed at OpenRouter (Gemini, GPT-OSS, GPT-5-mini)."""

    def __init__(self, spec, **_):
        super().__init__(spec)
        from openai import OpenAI
        key = os.environ.get(config.OPENROUTER_API_KEY_ENV)
        if not key:
            raise RuntimeError(f"Set {config.OPENROUTER_API_KEY_ENV} for {spec.key}")
        self.client = OpenAI(api_key=key, base_url=config.OPENROUTER_BASE_URL)

    def _extra_body(self) -> dict:
        # Disable hidden reasoning where the provider supports it (Section B.1:
        # "we set thinking to be false via the API"). OpenRouter passes this
        # through to Gemini/GPT. Pro/GPT-5.2 may still emit hidden reasoning.
        if config.DISABLE_THINKING and self.spec.family in {"gemini", "gpt"}:
            return {"reasoning": {"enabled": False}}
        return {}

    def generate(self, messages, params=None, prefill=None) -> str:
        params = params or GenerationParams()
        payload = [{"role": m["role"], "content": m["content"]} for m in messages]
        if prefill:
            # Best-effort prefill via a trailing assistant message.
            payload.append({"role": "assistant", "content": prefill})

        def call():
            return self.client.chat.completions.create(
                model=self.spec.model_id,
                messages=payload,
                temperature=params.temperature,
                top_p=params.top_p,
                max_tokens=params.max_new_tokens,
                stop=list(params.stop) or None,
                extra_body=self._extra_body(),
            )

        resp = self._with_retries(call)
        return (resp.choices[0].message.content or "").strip()

    def generate_batch(self, batch, params=None, prefills=None) -> list[str]:
        prefills = prefills or [None] * len(batch)
        with ThreadPoolExecutor(max_workers=config.API_CONCURRENCY) as ex:
            return list(ex.map(
                lambda a: self.generate(a[0], params, a[1]), zip(batch, prefills)
            ))
