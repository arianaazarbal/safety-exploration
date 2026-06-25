"""OpenAI API backend — used for the GPT-5-mini validation judge (Section 2.1)."""
from __future__ import annotations

from typing import Sequence

from emoinstab.config import ModelSpec
from emoinstab.models._api_common import require_env, threaded_map, with_retry
from emoinstab.models.base import Conversation, ModelClient, SamplingParams


class OpenAIClient(ModelClient):
    def __init__(self, spec: ModelSpec):
        super().__init__(spec)
        from openai import OpenAI

        self._client = OpenAI(api_key=require_env("OPENAI_API_KEY"))

    @with_retry
    def _once(self, messages: list[dict], params: SamplingParams) -> str:
        resp = self._client.chat.completions.create(
            model=self.spec.model_id,
            messages=messages,
            temperature=params.temperature,
            max_tokens=params.max_tokens,
        )
        return resp.choices[0].message.content or ""

    def chat(self, messages: Conversation, params: SamplingParams | None = None) -> list[str]:
        params = params or self.default_params()
        payload = [m.as_dict() for m in messages]
        return [self._once(payload, params) for _ in range(params.n)]

    def chat_batch(
        self, conversations: Sequence[Conversation], params: SamplingParams | None = None
    ) -> list[list[str]]:
        params = params or self.default_params()
        return threaded_map(lambda c: self.chat(c, params), list(conversations))
