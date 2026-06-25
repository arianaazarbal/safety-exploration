"""OpenRouter backend (OpenAI-compatible) — the paper's access path for Gemini.

The paper accessed Gemini (and other closed models) via OpenRouter ids such as
``google/gemini-2.5-flash``. We reuse the OpenAI SDK pointed at the OpenRouter
base URL. ``thinking=false`` is requested via ``extra_body.reasoning`` where the
provider honours it; note (per Appendix B.1) Gemini 2.5 Pro may still emit hidden
reasoning regardless.
"""
from __future__ import annotations

from typing import Sequence

from emoinstab.config import ModelSpec
from emoinstab.models._api_common import require_env, threaded_map, with_retry
from emoinstab.models.base import Conversation, ModelClient, SamplingParams


class OpenRouterClient(ModelClient):
    BASE_URL = "https://openrouter.ai/api/v1"

    def __init__(self, spec: ModelSpec):
        super().__init__(spec)
        from openai import OpenAI

        self._client = OpenAI(
            api_key=require_env("OPENROUTER_API_KEY"), base_url=self.BASE_URL
        )

    @with_retry
    def _once(self, messages: list[dict], params: SamplingParams) -> str:
        extra_body = {}
        if not params.thinking:
            # OpenRouter unifies reasoning control under `reasoning`.
            extra_body["reasoning"] = {"enabled": False}
        resp = self._client.chat.completions.create(
            model=self.spec.model_id,
            messages=messages,
            temperature=params.temperature,
            max_tokens=params.max_tokens,
            top_p=params.top_p,
            extra_body=extra_body or None,
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
