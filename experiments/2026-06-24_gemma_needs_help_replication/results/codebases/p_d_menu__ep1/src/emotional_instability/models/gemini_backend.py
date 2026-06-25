"""Gemini subject backend via OpenRouter (Appendix B.1).

The paper accesses Gemini through OpenRouter (`google/gemini-2.5-flash`,
`google/gemini-2.5-pro`) with thinking disabled. OpenRouter is OpenAI-compatible,
so we use the `openai` SDK pointed at the OpenRouter base URL.

Notes / faithful caveats:
  * Thinking is disabled via `extra_body={"reasoning": {"enabled": False}}`,
    but the paper notes Gemini-2.5-Pro may still emit hidden reasoning (B.1).
  * Gemini has no publicly accessible base model, so the Section 3 prefill
    experiment is not run on Gemini (a paper limitation). `continue_text` is a
    best-effort approximation: it appends the prefill as an assistant turn and
    asks the model to continue. Most chat APIs do not truly continue an
    assistant message, so this is flagged in `meta`.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

from .base import GenerationResult, Message

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


@dataclass
class OpenRouterBackend:
    name: str
    api_id: str
    thinking: bool = False
    is_chat: bool = True

    def __post_init__(self) -> None:
        self.supports_chat = True
        self.supports_prefill = False  # not truly supported over the API
        self._client = None

    def _ensure_client(self):
        if self._client is None:
            from openai import OpenAI

            self._client = OpenAI(
                base_url=OPENROUTER_BASE_URL,
                api_key=os.environ["OPENROUTER_API_KEY"],
            )
        return self._client

    def _extra_body(self) -> dict:
        # Disable provider-side reasoning/thinking where supported.
        if not self.thinking:
            return {"reasoning": {"enabled": False}}
        return {}

    def generate(
        self,
        messages: list[Message],
        *,
        temperature: float = 1.0,
        max_new_tokens: int = 2048,
        seed: int | None = None,
        stop: list[str] | None = None,
    ) -> GenerationResult:
        client = self._ensure_client()
        resp = client.chat.completions.create(
            model=self.api_id,
            messages=list(messages),
            temperature=temperature,
            max_tokens=max_new_tokens,
            seed=seed,
            stop=stop,
            extra_body=self._extra_body(),
        )
        choice = resp.choices[0]
        return GenerationResult(
            text=choice.message.content or "",
            n_new_tokens=getattr(resp.usage, "completion_tokens", None),
            meta={"backend": "openrouter", "finish_reason": choice.finish_reason},
        )

    def continue_text(
        self,
        messages: list[Message],
        prefill: str,
        *,
        temperature: float = 1.0,
        max_new_tokens: int = 2048,
        seed: int | None = None,
    ) -> GenerationResult:
        # Best-effort: include the prefill as a partial assistant turn. Some
        # OpenRouter providers honour a trailing assistant message as a prefix.
        msgs = list(messages) + [{"role": "assistant", "content": prefill}]
        client = self._ensure_client()
        resp = client.chat.completions.create(
            model=self.api_id,
            messages=msgs,
            temperature=temperature,
            max_tokens=max_new_tokens,
            seed=seed,
            extra_body=self._extra_body(),
        )
        choice = resp.choices[0]
        return GenerationResult(
            text=choice.message.content or "",
            n_new_tokens=getattr(resp.usage, "completion_tokens", None),
            meta={
                "backend": "openrouter",
                "prefill_is_approximate": True,
                "finish_reason": choice.finish_reason,
            },
        )
