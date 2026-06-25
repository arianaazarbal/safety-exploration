"""API-backed clients: OpenAI-compatible (OpenRouter -> Gemini, GPT-5-mini) and Anthropic.

Used for Gemini targets and for all judge / auditor / onset / paraphrase roles.
"""
from __future__ import annotations

import os
from typing import Any

from tenacity import retry, stop_after_attempt, wait_exponential

from .base import Conversation, GenParams, Message, ModelClient, ModelSpec


def _split_system(conversation: Conversation) -> tuple[str | None, list[dict[str, str]]]:
    system = None
    msgs: list[dict[str, str]] = []
    for m in conversation:
        if m.role == "system":
            # concatenate multiple system messages if present
            system = m.content if system is None else system + "\n\n" + m.content
        else:
            msgs.append({"role": m.role, "content": m.content})
    return system, msgs


class OpenAICompatClient(ModelClient):
    """OpenAI Chat Completions API. Points at OpenRouter by default (Gemini, GPT-5-mini)."""

    def __init__(self, spec: ModelSpec):
        super().__init__(spec)
        from openai import OpenAI  # imported lazily

        api_key = os.environ.get(spec.api_key_env or "OPENROUTER_API_KEY")
        if not api_key:
            raise RuntimeError(
                f"{spec.api_key_env or 'OPENROUTER_API_KEY'} is required for {spec.name}."
            )
        self.client = OpenAI(api_key=api_key, base_url=spec.api_base)
        self.model_id = spec.model_id or spec.name

    def _extra_body(self) -> dict[str, Any]:
        body: dict[str, Any] = {}
        if self.spec.disable_thinking:
            # OpenRouter passes provider-specific reasoning controls through `reasoning`.
            # For Gemini, effort "none" / a zero budget disables visible reasoning.
            body["reasoning"] = {"enabled": False}
        return body

    @retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, max=30))
    def _call(self, messages: list[dict[str, str]], params: GenParams, n: int) -> list[str]:
        kwargs: dict[str, Any] = dict(
            model=self.model_id,
            messages=messages,
            temperature=params.temperature,
            top_p=params.top_p,
            max_tokens=params.max_new_tokens,
            n=n,
        )
        if params.stop:
            kwargs["stop"] = params.stop
        if params.seed is not None:
            kwargs["seed"] = params.seed
        extra = self._extra_body()
        if extra:
            kwargs["extra_body"] = extra
        resp = self.client.chat.completions.create(**kwargs)
        return [c.message.content or "" for c in resp.choices]

    def generate_chat(self, conversation: Conversation, params: GenParams) -> list[str]:
        system, msgs = _split_system(conversation)
        if system is not None:
            msgs = [{"role": "system", "content": system}] + msgs
        # Some providers cap `n`; request in one shot but fall back to a loop on error.
        try:
            return self._call(msgs, params, params.n)
        except Exception:
            return [self._call(msgs, params, 1)[0] for _ in range(params.n)]


class AnthropicClient(ModelClient):
    """Anthropic Messages API. Used for the frustration judge, Petri auditor/judge,
    onset labeller, and paraphraser."""

    def __init__(self, spec: ModelSpec):
        super().__init__(spec)
        import anthropic  # lazy import

        api_key = os.environ.get(spec.api_key_env or "ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError(
                f"{spec.api_key_env or 'ANTHROPIC_API_KEY'} is required for {spec.name}."
            )
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model_id = spec.model_id or spec.name

    @retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, max=30))
    def _call(self, system: str | None, msgs: list[dict[str, str]], params: GenParams) -> str:
        kwargs: dict[str, Any] = dict(
            model=self.model_id,
            messages=msgs,
            max_tokens=params.max_new_tokens,
            temperature=params.temperature,
            top_p=params.top_p,
        )
        if system:
            kwargs["system"] = system
        if params.stop:
            kwargs["stop_sequences"] = params.stop
        resp = self.client.messages.create(**kwargs)
        return "".join(block.text for block in resp.content if block.type == "text")

    def generate_chat(self, conversation: Conversation, params: GenParams) -> list[str]:
        system, msgs = _split_system(conversation)
        return [self._call(system, msgs, params) for _ in range(params.n)]
