"""Gemini (and any OpenRouter-hosted model) via the OpenAI-compatible API.

The paper reaches Gemini through OpenRouter and sets "thinking to be false via
the API" (Appendix B), while noting Gemini-2.5-Pro may still emit hidden
reasoning. We mirror both: request ``reasoning.enabled=false`` and, as a
belt-and-braces measure, ``extra_body`` to disable Google thinking budgets.
"""

from __future__ import annotations

from ..config import ModelSpec
from .._version_compat import OPENROUTER_BASE_URL
from ..utils import stable_hash
from ._api_common import cached_call, require_env
from .base import GenConfig, GenResult, Message, ModelProvider


class OpenRouterProvider(ModelProvider):
    def __init__(self, spec: ModelSpec, *, use_cache: bool = True):
        super().__init__(spec)
        from openai import OpenAI

        self.use_cache = use_cache
        self.client = OpenAI(
            base_url=OPENROUTER_BASE_URL,
            api_key=require_env("OPENROUTER_API_KEY"),
        )

    def _generate(
        self, messages: list[Message], gen: GenConfig, prefill: str | None
    ) -> GenResult:
        if prefill is not None:
            raise NotImplementedError("OpenRouter (closed Gemini) does not support prefill")

        payload_messages = [m.to_dict() for m in messages]
        # extra_body disables provider-side reasoning where the upstream supports it.
        extra_body = {
            "reasoning": {"enabled": False},
            "provider": {"require_parameters": False},
        }
        request = dict(
            model=self.spec.model_id,
            messages=payload_messages,
            temperature=gen.temperature,
            top_p=gen.top_p,
            max_tokens=gen.max_new_tokens,
            stop=list(gen.stop) if gen.stop else None,
        )

        cache_payload = {
            "provider": "openrouter",
            "request": request,
            "extra_body": extra_body,
            "sample_index": gen.sample_index,
            "thinking": gen.thinking,
        }

        def _call() -> str:
            resp = self.client.chat.completions.create(**request, extra_body=extra_body)
            return resp.choices[0].message.content or ""

        text = cached_call(cache_payload, _call, use_cache=self.use_cache)
        return GenResult(text=text, meta={"cache_key": stable_hash(cache_payload)})
