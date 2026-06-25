"""OpenRouter backend (OpenAI-compatible) for Gemini target models.

The paper accesses closed models through OpenRouter and disables thinking via the
API ("we set thinking to be false via the API. However, Gemini-2.5 Pro and
GPT-5.2 Chat may produce hidden reasoning that is not prevented by this
setting"). We mirror that: pass a reasoning/​thinking-disable hint, but do not
assume it is fully honoured for Pro.
"""
from __future__ import annotations

import os

from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_random_exponential

from .base import ChatMessage, GenerationResult, ModelClient, SamplingParams

_BASE_URL = "https://openrouter.ai/api/v1"


class OpenRouterClient(ModelClient):
    def __init__(self, name: str, api_id: str, api_key: str | None = None):
        self.name = name
        self.api_id = api_id
        self._client = OpenAI(
            base_url=_BASE_URL,
            api_key=api_key or os.environ["OPENROUTER_API_KEY"],
        )

    def _extra_body(self, params: SamplingParams) -> dict:
        body: dict = {}
        if not params.thinking:
            # OpenRouter forwards provider-specific reasoning controls. Setting a
            # zero/low reasoning budget is the closest portable analogue to the
            # paper's "thinking=false". Honoured by Gemini Flash; Pro may ignore.
            body["reasoning"] = {"enabled": False}
        return body

    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
    def generate(self, messages: list[ChatMessage], params: SamplingParams) -> GenerationResult:
        resp = self._client.chat.completions.create(
            model=self.api_id,
            messages=[m.as_dict() for m in messages],
            temperature=params.temperature,
            max_tokens=params.max_tokens,
            top_p=params.top_p,
            stop=params.stop or None,
            seed=params.seed,
            extra_body=self._extra_body(params),
        )
        choice = resp.choices[0]
        usage = resp.usage
        return GenerationResult(
            text=choice.message.content or "",
            prompt_tokens=getattr(usage, "prompt_tokens", None),
            completion_tokens=getattr(usage, "completion_tokens", None),
            finish_reason=choice.finish_reason,
            raw=resp,
        )

    def generate_batch(
        self, conversations: list[list[ChatMessage]], params: SamplingParams
    ) -> list[GenerationResult]:
        # API batching is handled by the caller's thread pool; keep this simple.
        return [self.generate(c, params) for c in conversations]
