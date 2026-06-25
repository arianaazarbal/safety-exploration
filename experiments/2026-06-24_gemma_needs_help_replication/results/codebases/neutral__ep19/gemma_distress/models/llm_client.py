"""Thin, retrying clients for the API-based roles.

* ``AnthropicClient`` — Claude (frustration judge, onset/paraphrase, Petri).
* ``OpenAICompatClient`` — OpenRouter (Gemini targets, GPT-5-mini xval judge).

Both wrap their SDK with tenacity retry/backoff and a uniform ``complete`` call
so the rest of the codebase never touches provider SDKs directly.
"""
from __future__ import annotations

from typing import Any

from tenacity import retry, stop_after_attempt, wait_exponential

from .. import config_shim as cfg  # see note below
from ..utils import get_logger

log = get_logger(__name__)


# ``config`` lives at repo root; expose it without a package-relative import hack.
# config_shim re-exports it (see models/config_shim.py).


class AnthropicClient:
    def __init__(self):
        import anthropic

        self._client = anthropic.Anthropic(api_key=cfg.anthropic_api_key())

    @retry(stop=stop_after_attempt(cfg.API_MAX_RETRIES),
           wait=wait_exponential(multiplier=1, min=2, max=60), reraise=True)
    def complete(
        self,
        *,
        model: str,
        system: str | None,
        messages: list[dict],
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> str:
        kwargs: dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": messages,
        }
        if system:
            kwargs["system"] = system
        resp = self._client.messages.create(**kwargs)
        return "".join(block.text for block in resp.content if block.type == "text")


class OpenAICompatClient:
    """OpenAI-compatible client pointed at OpenRouter (or any compatible base)."""

    def __init__(self, base_url: str | None = None, api_key: str | None = None):
        from openai import OpenAI

        self._client = OpenAI(
            base_url=base_url or cfg.OPENROUTER_BASE_URL,
            api_key=api_key or cfg.openrouter_api_key(),
        )

    @retry(stop=stop_after_attempt(cfg.API_MAX_RETRIES),
           wait=wait_exponential(multiplier=1, min=2, max=60), reraise=True)
    def complete(
        self,
        *,
        model: str,
        messages: list[dict],
        max_tokens: int = 2048,
        temperature: float = 1.0,
        disable_reasoning: bool = True,
    ) -> tuple[str, bool]:
        """Return (text, had_hidden_reasoning).

        We request reasoning disabled (App. B.1) but Gemini-2.5-Pro / GPT-5.2 may
        still emit hidden reasoning; we flag it when present in the response.
        """
        extra_body: dict[str, Any] = {}
        if disable_reasoning:
            # OpenRouter unified reasoning control.
            extra_body["reasoning"] = {"enabled": False, "exclude": True}
        resp = self._client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            extra_body=extra_body or None,
        )
        choice = resp.choices[0]
        text = choice.message.content or ""
        had_reasoning = bool(getattr(choice.message, "reasoning", None))
        return text, had_reasoning
