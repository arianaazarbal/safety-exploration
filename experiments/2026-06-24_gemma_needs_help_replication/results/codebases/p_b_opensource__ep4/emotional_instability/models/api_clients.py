"""API backends used for judging and open-ended auditing.

* `AnthropicBackend` — Claude Sonnet 4 (frustration judge, onset labeller,
  paraphraser, Petri auditor) and Claude Opus 4 (Petri judge).
* `OpenAIBackend` — GPT-5-mini, used only for the judge-agreement validation
  (Section 2.1) and optionally as a non-Claude target elsewhere.

Both expose the same `ModelBackend.generate` interface so judging code does not
care which provider scores a response. A system message, if present, is routed
to the provider's dedicated system field.
"""

from __future__ import annotations

import os

from tenacity import retry, stop_after_attempt, wait_exponential

from ..config import MAX_NEW_TOKENS, ModelSpec
from ._concurrency import threaded_map
from .base import ChatMessage, GenerationResult, ModelBackend, SamplingParams


def _split_system(messages: list[ChatMessage]) -> tuple[str, list[ChatMessage]]:
    system = "\n\n".join(m.content for m in messages if m.role == "system")
    rest = [m for m in messages if m.role != "system"]
    return system, rest


class AnthropicBackend(ModelBackend):
    supports_prefill = True  # Anthropic supports assistant prefill.
    max_workers = 8

    def __init__(self, spec: ModelSpec):
        self.spec = spec
        self._client = None

    def generate_batch(self, batch, params):
        return threaded_map(lambda m: self.generate(m, params), batch, self.max_workers)

    @property
    def client(self):
        if self._client is None:
            import anthropic

            api_key = os.environ.get("ANTHROPIC_API_KEY")
            if not api_key:
                raise RuntimeError("ANTHROPIC_API_KEY is not set")
            self._client = anthropic.Anthropic(api_key=api_key)
        return self._client

    @retry(
        stop=stop_after_attempt(6),
        wait=wait_exponential(multiplier=1, min=2, max=60),
        reraise=True,
    )
    def generate(
        self, messages: list[ChatMessage], params: SamplingParams
    ) -> GenerationResult:
        system, rest = _split_system(messages)
        resp = self.client.messages.create(
            model=self.spec.model_id,
            system=system or None,
            messages=[m.as_dict() for m in rest],
            temperature=params.temperature,
            max_tokens=params.max_new_tokens or MAX_NEW_TOKENS,
        )
        text = "".join(b.text for b in resp.content if getattr(b, "type", None) == "text")
        return GenerationResult(text=text.strip(), finish_reason=resp.stop_reason)

    def generate_with_prefill(
        self, messages: list[ChatMessage], prefill: str, params: SamplingParams
    ) -> GenerationResult:
        system, rest = _split_system(messages)
        msgs = [m.as_dict() for m in rest] + [{"role": "assistant", "content": prefill}]
        resp = self.client.messages.create(
            model=self.spec.model_id,
            system=system or None,
            messages=msgs,
            temperature=params.temperature,
            max_tokens=params.max_new_tokens or MAX_NEW_TOKENS,
        )
        text = "".join(b.text for b in resp.content if getattr(b, "type", None) == "text")
        return GenerationResult(text=text.strip(), prefill=prefill, finish_reason=resp.stop_reason)


class OpenAIBackend(ModelBackend):
    supports_prefill = False
    max_workers = 8

    def __init__(self, spec: ModelSpec):
        self.spec = spec
        self._client = None

    def generate_batch(self, batch, params):
        return threaded_map(lambda m: self.generate(m, params), batch, self.max_workers)

    @property
    def client(self):
        if self._client is None:
            from openai import OpenAI

            api_key = os.environ.get("OPENAI_API_KEY")
            if not api_key:
                raise RuntimeError("OPENAI_API_KEY is not set")
            self._client = OpenAI(api_key=api_key)
        return self._client

    @retry(
        stop=stop_after_attempt(6),
        wait=wait_exponential(multiplier=1, min=2, max=60),
        reraise=True,
    )
    def generate(
        self, messages: list[ChatMessage], params: SamplingParams
    ) -> GenerationResult:
        # GPT-5-mini is a reasoning model: it uses max_completion_tokens and a
        # fixed temperature. We keep the call minimal and provider-tolerant.
        kwargs = dict(
            model=self.spec.model_id,
            messages=[m.as_dict() for m in messages],
            max_completion_tokens=params.max_new_tokens or MAX_NEW_TOKENS,
        )
        resp = self.client.chat.completions.create(**kwargs)
        choice = resp.choices[0]
        return GenerationResult(
            text=(choice.message.content or "").strip(),
            finish_reason=choice.finish_reason,
        )
