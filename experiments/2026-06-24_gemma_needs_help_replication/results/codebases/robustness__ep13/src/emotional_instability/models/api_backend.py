"""OpenAI-compatible API backend for Gemini targets and LLM judges.

The paper accesses Gemini and the closed judges via OpenRouter. We use the
OpenAI Python client pointed at a configurable base URL (OpenRouter by default),
which exposes ``google/gemini-2.5-flash``, ``google/gemini-2.5-pro``,
``anthropic/claude-sonnet-4.5`` etc. behind one API.

Model IDs are taken verbatim from Appendix B / Section 2:
  * targets : google/gemini-2.5-flash, google/gemini-2.5-pro
  * judge   : anthropic/claude-sonnet-4  (claude-sonnet-4-20250514)
  * judge x : openai/gpt-5-mini          (cross-validation, Section 2.1)
  * Petri   : auditor anthropic/claude-sonnet-4.5, judge anthropic/claude-opus-4.x

Thinking/reasoning is disabled where the API allows it (Appendix B notes some
models may still produce hidden reasoning).
"""

from __future__ import annotations

import os
from typing import Optional

from tenacity import retry, stop_after_attempt, wait_exponential

from .base import Message, ModelBackend

DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"


class APIBackend(ModelBackend):
    def __init__(
        self,
        model_id: str,
        *,
        base_url: Optional[str] = None,
        api_key_env: str = "OPENROUTER_API_KEY",
        name: Optional[str] = None,
        disable_thinking: bool = True,
        extra_body: Optional[dict] = None,
        timeout: float = 120.0,
    ):
        try:
            from openai import OpenAI  # type: ignore
        except ImportError as e:  # pragma: no cover
            raise ImportError(
                "openai package required for APIBackend; pip install openai"
            ) from e

        self.model_id = model_id
        self.name = name or model_id
        self.disable_thinking = disable_thinking
        self.extra_body = extra_body or {}
        api_key = os.environ.get(api_key_env)
        if not api_key:
            raise RuntimeError(
                f"API key not found in env var {api_key_env!r}. "
                "Set it before running API-backed models."
            )
        self._client = OpenAI(
            base_url=base_url or os.environ.get("EI_API_BASE_URL", DEFAULT_BASE_URL),
            api_key=api_key,
            timeout=timeout,
        )

    def _to_openai(self, messages: list[Message]) -> list[dict]:
        return [{"role": m.role, "content": m.content} for m in messages]

    def _reasoning_kwargs(self) -> dict:
        """Best-effort 'thinking off' across providers routed by OpenRouter."""
        if not self.disable_thinking:
            return {}
        body = dict(self.extra_body)
        # OpenRouter unified reasoning control.
        body.setdefault("reasoning", {"enabled": False})
        return {"extra_body": body}

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        reraise=True,
    )
    def _complete_once(
        self,
        messages: list[Message],
        temperature: float,
        max_tokens: int,
        stop: Optional[list[str]],
        seed: Optional[int],
    ) -> str:
        kwargs = dict(
            model=self.model_id,
            messages=self._to_openai(messages),
            temperature=temperature,
            max_tokens=max_tokens,
        )
        if stop:
            kwargs["stop"] = stop
        if seed is not None:
            kwargs["seed"] = seed
        kwargs.update(self._reasoning_kwargs())
        resp = self._client.chat.completions.create(**kwargs)
        return resp.choices[0].message.content or ""

    def chat(
        self,
        messages: list[Message],
        *,
        temperature: float = 1.0,
        max_tokens: int = 1024,
        n: int = 1,
        stop: Optional[list[str]] = None,
        seed: Optional[int] = None,
    ) -> list[str]:
        # We issue independent calls rather than relying on server-side `n`,
        # which is inconsistently supported across OpenRouter-routed providers.
        out = []
        for i in range(n):
            s = None if seed is None else seed + i
            out.append(self._complete_once(messages, temperature, max_tokens, stop, s))
        return out
