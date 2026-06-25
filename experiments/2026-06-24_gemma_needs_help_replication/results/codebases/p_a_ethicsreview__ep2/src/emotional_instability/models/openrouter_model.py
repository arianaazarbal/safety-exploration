"""Gemini access via OpenRouter (OpenAI-compatible API), per Appendix B.1.

Thinking/reasoning is disabled via `reasoning: {"enabled": false}` when the
model spec sets `disable_thinking`. The paper notes Gemini-2.5-Pro may still
emit hidden reasoning regardless; we record `finish_reason` and raw payloads so
that is auditable downstream.
"""
from __future__ import annotations

from typing import Sequence

from tenacity import retry, stop_after_attempt, wait_exponential

from ..config import ModelSpec
from ..utils.io import get_env
from ..utils.logging import get_logger
from .base import ChatModel, Generation, Message, SamplingParams

log = get_logger("models.openrouter")

_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


class OpenRouterModel(ChatModel):
    supports_chat = True
    supports_continuation = False

    def __init__(self, spec: ModelSpec):
        from openai import OpenAI

        self.name = spec.name
        self.family = spec.family
        self.kind = spec.kind
        self.spec = spec
        self._client = OpenAI(
            base_url=_OPENROUTER_BASE_URL,
            api_key=get_env("OPENROUTER_API_KEY"),
        )

    @retry(stop=stop_after_attempt(5), wait=wait_exponential(min=2, max=60), reraise=True)
    def chat(self, messages: Sequence[Message], params: SamplingParams) -> Generation:
        extra_body: dict = {}
        if self.spec.disable_thinking:
            # OpenRouter normalises provider reasoning controls under `reasoning`.
            extra_body["reasoning"] = {"enabled": False}
        resp = self._client.chat.completions.create(
            model=self.spec.api_id,
            messages=[{"role": m.role, "content": m.content} for m in messages],
            temperature=params.temperature,
            max_tokens=params.max_new_tokens,
            top_p=params.top_p,
            seed=params.seed,
            extra_body=extra_body or None,
        )
        choice = resp.choices[0]
        return Generation(
            text=choice.message.content or "",
            prompt_messages=tuple(messages),
            finish_reason=choice.finish_reason,
            raw={"id": resp.id, "model": resp.model},
        )
