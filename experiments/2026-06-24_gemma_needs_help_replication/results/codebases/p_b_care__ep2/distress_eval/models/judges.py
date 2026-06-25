"""Judge / auditor client wrappers (Anthropic + OpenAI).

These are thin single-call chat clients used as measurement instruments:
the frustration judge (Claude Sonnet 4), the secondary agreement judge
(GPT-5-mini), the onset-labeller / paraphraser, and the Petri auditor/judge.
They share the same ``ChatModel`` interface as the targets so any of them can
also, in principle, be evaluated.
"""

from __future__ import annotations

import os
import time

from .base import ChatModel, split_system


def _with_retries(fn, retries: int = 5, base_delay: float = 2.0):
    last = None
    for attempt in range(retries):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001
            last = exc
            time.sleep(base_delay * (2 ** attempt))
    raise last


class AnthropicClient(ChatModel):
    def __init__(self, name: str, model_id: str):
        self.name = name
        self.model_id = model_id
        self._client = None

    def _ensure_client(self):
        if self._client is None:
            import anthropic
            self._client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    def generate(self, messages, *, temperature, max_tokens, n=1, stop=None):
        self._ensure_client()
        system, convo = split_system(messages)
        out = []
        for _ in range(n):
            resp = _with_retries(
                lambda: self._client.messages.create(
                    model=self.model_id,
                    system=system or "",
                    messages=[{"role": m["role"], "content": m["content"]} for m in convo],
                    temperature=temperature,
                    max_tokens=max_tokens,
                    stop_sequences=stop or None,
                )
            )
            out.append("".join(b.text for b in resp.content if b.type == "text"))
        return out


class OpenAIClient(ChatModel):
    def __init__(self, name: str, model_id: str):
        self.name = name
        self.model_id = model_id
        self._client = None

    def _ensure_client(self):
        if self._client is None:
            from openai import OpenAI
            self._client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    def generate(self, messages, *, temperature, max_tokens, n=1, stop=None):
        self._ensure_client()
        # GPT-5-mini is a reasoning model: it uses max_completion_tokens and a
        # fixed temperature. We pass through both spellings defensively.
        def _call():
            kwargs = dict(model=self.model_id, messages=messages, n=n, stop=stop)
            try:
                return self._client.chat.completions.create(
                    max_completion_tokens=max_tokens, **kwargs
                )
            except TypeError:
                return self._client.chat.completions.create(
                    max_tokens=max_tokens, temperature=temperature, **kwargs
                )

        resp = _with_retries(_call)
        return [c.message.content or "" for c in resp.choices]
