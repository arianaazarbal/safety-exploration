"""API backends: Gemini via OpenRouter, and Claude via the Anthropic SDK.

* :class:`OpenRouterChatModel` — Gemini-2.5-{flash,pro} targets (Section 2). The
  paper accessed Gemini through OpenRouter with thinking disabled. OpenRouter is
  OpenAI-API-compatible, so we use the ``openai`` SDK pointed at its base URL and
  pass ``reasoning={"enabled": False}`` (best-effort thinking disable).
* :class:`AnthropicChatModel` — Claude judge / Petri auditor & judge / onset &
  paraphrase helpers. Model ids are pinned by the paper (claude-sonnet-4-20250514,
  claude-opus-4-20250514) for replication fidelity.

API models do not support assistant-prefill continuation in a way that matches
local logit access, so :meth:`continue_prefill` raises. This is why Section 3 is
Gemma-only (the paper makes the same caveat for closed Gemini).
"""
from __future__ import annotations

import os
import time
from typing import Optional

from ..config import ModelSpec, SamplingConfig
from .base import ChatModel, Message

_RETRIES = 5
_BACKOFF = 2.0


def _with_retries(fn):
    last = None
    for attempt in range(_RETRIES):
        try:
            return fn()
        except Exception as e:  # noqa: BLE001 - surface after retries
            last = e
            time.sleep(_BACKOFF * (attempt + 1))
    raise last


class OpenRouterChatModel(ChatModel):
    def __init__(self, spec: ModelSpec, *, api_key: Optional[str] = None):
        super().__init__(spec)
        from openai import OpenAI

        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key or os.environ.get("OPENROUTER_API_KEY", ""),
        )

    def _split_system(self, messages: list[Message]):
        # OpenAI schema keeps system as a normal message; pass through.
        return list(messages)

    def generate(self, messages, sampling, n=1):
        extra_body = {}
        if self.spec.disable_thinking:
            # OpenRouter reasoning control; ignored by providers that lack it.
            extra_body["reasoning"] = {"enabled": False}

        def _call():
            return self.client.chat.completions.create(
                model=self.spec.model_id,
                messages=self._split_system(messages),
                temperature=sampling.temperature,
                top_p=sampling.top_p,
                max_tokens=sampling.max_new_tokens,
                n=n,
                extra_body=extra_body or None,
            )

        resp = _with_retries(_call)
        return [c.message.content or "" for c in resp.choices]

    def continue_prefill(self, messages, prefill, sampling, n=1):
        raise NotImplementedError(
            "Gemini (API) does not support assistant prefill continuation; "
            "Section 3 is Gemma-only by design."
        )


class AnthropicChatModel(ChatModel):
    """Thin wrapper used for judge/auditor roles (not an evaluation target)."""

    def __init__(self, spec: ModelSpec, *, api_key: Optional[str] = None):
        super().__init__(spec)
        import anthropic

        self.client = anthropic.Anthropic(
            api_key=api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        )

    def generate(self, messages, sampling, n=1):
        system = None
        conv = []
        for m in messages:
            if m["role"] == "system":
                system = (system + "\n\n" + m["content"]) if system else m["content"]
            else:
                conv.append({"role": m["role"], "content": m["content"]})

        out = []
        for _ in range(n):

            def _call():
                kwargs = dict(
                    model=self.spec.model_id,
                    max_tokens=sampling.max_new_tokens,
                    temperature=sampling.temperature,
                    messages=conv,
                )
                if system:
                    kwargs["system"] = system
                return self.client.messages.create(**kwargs)

            resp = _with_retries(_call)
            text = "".join(
                block.text for block in resp.content if getattr(block, "type", None) == "text"
            )
            out.append(text)
        return out

    def continue_prefill(self, messages, prefill, sampling, n=1):
        # Anthropic supports assistant-prefill as the last message, but we never
        # use Claude as a prefill *target* in this replication.
        raise NotImplementedError
