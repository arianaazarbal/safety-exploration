"""OpenRouter-backed client (OpenAI-compatible).

Used for:
  * Gemini participants (closed-source, API-only)
  * Gemma participants when sampling cheaply in the cloud (Section 2)
  * All measurement instruments (Claude judge, GPT validation judge,
    Claude auditor, Claude-Opus Petri judge)

OpenRouter exposes an OpenAI-compatible /chat/completions endpoint, so we reuse
the `openai` SDK pointed at the OpenRouter base URL.
"""
from __future__ import annotations

from typing import Any

from tenacity import retry, stop_after_attempt, wait_exponential_jitter

from ..config import get_api_key
from .base import ChatMessage, GenerationResult, ModelClient, SamplingParams

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


class OpenRouterClient(ModelClient):
    def __init__(self, model_name: str, openrouter_id: str, thinking: bool = False):
        super().__init__(model_name)
        self.openrouter_id = openrouter_id
        self.thinking = thinking
        self._client = None

    @property
    def client(self):
        # Lazy import so the package imports without the SDK installed.
        if self._client is None:
            from openai import OpenAI

            self._client = OpenAI(
                base_url=OPENROUTER_BASE_URL,
                api_key=get_api_key("OPENROUTER_API_KEY"),
            )
        return self._client

    def _build_messages(
        self, messages: list[ChatMessage], params: SamplingParams
    ) -> list[dict[str, str]]:
        payload = [m.to_openai() for m in messages]
        # Prefill: append a trailing assistant message. OpenRouter forwards this to
        # providers that support assistant-prefix continuation (e.g. Gemma via
        # certain providers). For providers that do not, the local backend should be
        # used instead -- see registry routing for Section 3.
        if params.prefill is not None:
            payload.append({"role": "assistant", "content": params.prefill})
        return payload

    def _extra_body(self) -> dict[str, Any]:
        # Disable provider-side reasoning where supported (paper: thinking=false).
        if self.thinking:
            return {}
        return {"reasoning": {"enabled": False}}

    @retry(
        reraise=True,
        stop=stop_after_attempt(5),
        wait=wait_exponential_jitter(initial=1, max=30),
    )
    def chat(self, messages: list[ChatMessage], params: SamplingParams) -> GenerationResult:
        payload = self._build_messages(messages, params)
        resp = self.client.chat.completions.create(
            model=self.openrouter_id,
            messages=payload,
            temperature=params.temperature,
            top_p=params.top_p,
            max_tokens=params.max_tokens,
            stop=params.stop,
            seed=params.seed,
            extra_body=self._extra_body(),
        )
        choice = resp.choices[0]
        text = choice.message.content or ""
        return GenerationResult(
            text=text,
            model=self.openrouter_id,
            finish_reason=choice.finish_reason,
            prefill=params.prefill or "",
            raw=resp.model_dump() if hasattr(resp, "model_dump") else {},
        )
