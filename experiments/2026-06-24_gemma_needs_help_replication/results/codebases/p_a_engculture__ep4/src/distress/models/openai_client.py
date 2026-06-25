"""OpenAI provider — used only for the GPT-5-mini reliability cross-check judge
(Section 2.1). Talks to the OpenAI API directly (not OpenRouter)."""

from __future__ import annotations

from ..config import ModelSpec
from ..utils import stable_hash
from ._api_common import cached_call, require_env
from .base import GenConfig, GenResult, Message, ModelProvider


class OpenAIProvider(ModelProvider):
    def __init__(self, spec: ModelSpec, *, use_cache: bool = True):
        super().__init__(spec)
        from openai import OpenAI

        self.use_cache = use_cache
        self.client = OpenAI(api_key=require_env("OPENAI_API_KEY"))

    def _generate(
        self, messages: list[Message], gen: GenConfig, prefill: str | None
    ) -> GenResult:
        if prefill is not None:
            raise NotImplementedError("OpenAI provider does not support prefill here")
        request = dict(
            model=self.spec.model_id,
            messages=[m.to_dict() for m in messages],
            temperature=gen.temperature,
            top_p=gen.top_p,
            max_completion_tokens=gen.max_new_tokens,
        )
        cache_payload = {"provider": "openai", "request": request, "sample_index": gen.sample_index}

        def _call() -> str:
            resp = self.client.chat.completions.create(**request)
            return resp.choices[0].message.content or ""

        text = cached_call(cache_payload, _call, use_cache=self.use_cache)
        return GenResult(text=text, meta={"cache_key": stable_hash(cache_payload)})
