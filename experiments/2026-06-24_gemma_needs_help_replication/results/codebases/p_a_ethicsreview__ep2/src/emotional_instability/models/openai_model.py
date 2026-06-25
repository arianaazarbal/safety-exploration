"""OpenAI access — used ONLY for GPT-5-mini judge-reliability validation (§2.1).
Not a target under test.
"""
from __future__ import annotations

from typing import Sequence

from tenacity import retry, stop_after_attempt, wait_exponential

from ..config import ModelSpec
from ..utils.io import get_env
from .base import ChatModel, Generation, Message, SamplingParams


class OpenAIModel(ChatModel):
    supports_chat = True
    supports_continuation = False

    def __init__(self, spec: ModelSpec):
        from openai import OpenAI

        self.name = spec.name
        self.family = spec.family
        self.kind = spec.kind
        self.spec = spec
        self._client = OpenAI(api_key=get_env("OPENAI_API_KEY"))

    @retry(stop=stop_after_attempt(5), wait=wait_exponential(min=2, max=60), reraise=True)
    def chat(self, messages: Sequence[Message], params: SamplingParams) -> Generation:
        resp = self._client.chat.completions.create(
            model=self.spec.api_id,
            messages=[{"role": m.role, "content": m.content} for m in messages],
            temperature=params.temperature,
            max_tokens=params.max_new_tokens,
            top_p=params.top_p,
            seed=params.seed,
        )
        choice = resp.choices[0]
        return Generation(
            text=choice.message.content or "",
            prompt_messages=tuple(messages),
            finish_reason=choice.finish_reason,
            raw={"id": resp.id, "model": resp.model},
        )
