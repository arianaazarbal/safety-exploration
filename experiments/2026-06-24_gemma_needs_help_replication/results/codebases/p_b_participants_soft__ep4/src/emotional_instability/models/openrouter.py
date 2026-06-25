"""OpenRouter-backed client (OpenAI-compatible API).

Used for:
  * Gemini participants (google/gemini-2.5-flash, google/gemini-2.5-pro)
  * the Claude-Sonnet-4 frustration judge / onset labeler / paraphraser / auditor
  * the Claude-Opus-4 Petri judge
  * the GPT-5-mini validation judge

The paper routes all API models through OpenRouter (Appendix B.1), so we do the
same for provenance parity. A single API key (OPENROUTER_API_KEY) covers every
hosted model. `thinking=False` is mapped to each provider's "no reasoning"
control on a best-effort basis (the paper notes Gemini-2.5-Pro / GPT-5.2 may
still emit hidden reasoning regardless).
"""
from __future__ import annotations

import os
from typing import Sequence

from tenacity import (retry, retry_if_exception_type, stop_after_attempt,
                      wait_random_exponential)

from .base import ChatMessage, GenerationConfig, ModelClient

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


class _RetryableError(Exception):
    pass


class OpenRouterClient(ModelClient):
    def __init__(self, name: str, api_id: str, *, thinking_default: bool | None = None):
        super().__init__(name)
        self.api_id = api_id
        self.thinking_default = thinking_default
        # Imported lazily so the package imports without the openai dependency.
        from openai import OpenAI

        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "OPENROUTER_API_KEY is not set; required for OpenRouter-backed "
                f"model '{name}' ({api_id})."
            )
        self._client = OpenAI(base_url=OPENROUTER_BASE_URL, api_key=api_key)

    # ------------------------------------------------------------------ #
    def _extra_body(self, cfg: GenerationConfig) -> dict:
        """Build provider-specific knobs, primarily to suppress reasoning."""
        thinking = cfg.thinking if cfg.thinking is not None else self.thinking_default
        body: dict = {}
        if thinking is False:
            # OpenRouter's unified reasoning control; providers that support it
            # (Gemini, GPT) will disable/minimise hidden reasoning.
            body["reasoning"] = {"enabled": False}
        return body

    @retry(
        retry=retry_if_exception_type(_RetryableError),
        wait=wait_random_exponential(min=2, max=60),
        stop=stop_after_attempt(6),
        reraise=True,
    )
    def _call(self, messages: list[dict], cfg: GenerationConfig) -> str:
        try:
            resp = self._client.chat.completions.create(
                model=self.api_id,
                messages=messages,
                temperature=cfg.temperature,
                top_p=cfg.top_p,
                max_tokens=cfg.max_new_tokens,
                seed=cfg.seed,
                stop=list(cfg.stop) if cfg.stop else None,
                extra_body=self._extra_body(cfg) or None,
            )
        except Exception as exc:  # noqa: BLE001 - normalise to retryable
            # Rate limits / transient 5xx -> retry; everything else bubbles up.
            msg = str(exc).lower()
            if any(s in msg for s in ("rate", "429", "500", "502", "503", "timeout", "overloaded")):
                raise _RetryableError(str(exc)) from exc
            raise
        choice = resp.choices[0]
        return (choice.message.content or "").strip()

    def generate(self, messages: Sequence[ChatMessage], cfg: GenerationConfig) -> str:
        return self._call([m.as_dict() for m in messages], cfg)
