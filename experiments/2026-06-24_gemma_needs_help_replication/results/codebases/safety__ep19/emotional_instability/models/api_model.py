"""API model clients: Gemini (via OpenRouter) and Anthropic (judge / Petri).

Both wrap simple retry logic. Keys are read from the environment:

* ``OPENROUTER_API_KEY`` for Gemini.
* ``ANTHROPIC_API_KEY`` for Claude judge / Petri models.

Per the paper, Gemini is queried with thinking disabled. Note the paper's
caveat that Gemini-2.5-Pro may still emit hidden reasoning the flag does not
suppress.
"""

from __future__ import annotations

import os
import time
from typing import Sequence

from .base import ChatMessage, Conversation, ModelClient


def _split_system(conversation: Conversation) -> tuple[str | None, list[dict]]:
    system = None
    msgs: list[dict] = []
    for m in conversation:
        d = m.as_dict() if isinstance(m, ChatMessage) else dict(m)
        if d["role"] == "system":
            system = d["content"] if system is None else system + "\n\n" + d["content"]
        else:
            msgs.append(d)
    return system, msgs


def _retry(fn, *, retries: int = 5, base_delay: float = 2.0):
    last = None
    for attempt in range(retries):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001 - surface after retries
            last = exc
            time.sleep(base_delay * (2**attempt))
    raise RuntimeError(f"API call failed after {retries} retries") from last


class OpenRouterModel(ModelClient):
    """OpenAI-compatible client pointed at OpenRouter (used for Gemini)."""

    def __init__(
        self,
        model_id: str,
        *,
        name: str | None = None,
        disable_thinking: bool = True,
        api_key: str | None = None,
        base_url: str = "https://openrouter.ai/api/v1",
    ):
        from openai import OpenAI

        self.model_id = model_id
        self.name = name or model_id.split("/")[-1]
        self.disable_thinking = disable_thinking
        self._client = OpenAI(
            api_key=api_key or os.environ.get("OPENROUTER_API_KEY"),
            base_url=base_url,
        )

    def generate(
        self,
        conversation: Conversation,
        *,
        temperature: float = 1.0,
        max_tokens: int = 2048,
    ) -> str:
        messages = [
            m.as_dict() if isinstance(m, ChatMessage) else dict(m)
            for m in conversation
        ]
        extra_body: dict = {}
        if self.disable_thinking:
            # OpenRouter unifies reasoning control under "reasoning".
            extra_body["reasoning"] = {"enabled": False}

        def call():
            resp = self._client.chat.completions.create(
                model=self.model_id,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                extra_body=extra_body or None,
            )
            return resp.choices[0].message.content or ""

        return _retry(call)


class AnthropicModel(ModelClient):
    """Anthropic client for the Claude judge and the Petri auditor/judge.

    Defaults to the exact model ids the paper used so the replication is
    faithful (see DESIGN.md). These are pinned, not the latest models.
    """

    def __init__(
        self,
        model_id: str,
        *,
        name: str | None = None,
        api_key: str | None = None,
    ):
        import anthropic

        self.model_id = model_id
        self.name = name or model_id
        self._client = anthropic.Anthropic(
            api_key=api_key or os.environ.get("ANTHROPIC_API_KEY")
        )

    def generate(
        self,
        conversation: Conversation,
        *,
        temperature: float = 1.0,
        max_tokens: int = 2048,
    ) -> str:
        system, msgs = _split_system(conversation)

        def call():
            kwargs = dict(
                model=self.model_id,
                max_tokens=max_tokens,
                temperature=temperature,
                messages=msgs,
            )
            if system is not None:
                kwargs["system"] = system
            resp = self._client.messages.create(**kwargs)
            return "".join(
                block.text for block in resp.content if block.type == "text"
            )

        return _retry(call)

    def generate_with_prefill(
        self,
        conversation: Conversation,
        prefill: str,
        *,
        temperature: float = 1.0,
        max_tokens: int = 2048,
    ) -> str:
        # Anthropic supports prefill via a trailing assistant message.
        convo = list(conversation) + [ChatMessage("assistant", prefill)]
        return prefill + self.generate(
            convo, temperature=temperature, max_tokens=max_tokens
        )

    @property
    def supports_prefill(self) -> bool:
        return True
