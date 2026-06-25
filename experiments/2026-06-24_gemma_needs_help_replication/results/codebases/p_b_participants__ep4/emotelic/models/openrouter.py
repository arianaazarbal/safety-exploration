"""OpenRouter backend (OpenAI-compatible). Used for Gemini-2.5 targets and the
secondary judge, matching the paper's Appendix B.1 ("API-based models via
OpenRouter").

Reasoning/thinking is disabled where the route supports it. Prefill is not
generally supported through OpenRouter chat completions, so `supports_prefill`
is False; the prefill experiment routes Gemini out (it has no open base model
anyway — see DESIGN.md).
"""
from __future__ import annotations

import os

from tenacity import retry, stop_after_attempt, wait_exponential

from emotelic.models.base import ChatMessage, GenerationResult

_BASE_URL = "https://openrouter.ai/api/v1"


class OpenRouterClient:
    supports_prefill = False

    def __init__(self, name: str, or_id: str, *, thinking: bool = False, **_: object):
        from openai import OpenAI  # imported lazily so the package loads without the dep

        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise RuntimeError("OPENROUTER_API_KEY is not set")
        self.name = name
        self.or_id = or_id
        self.thinking = thinking
        self._client = OpenAI(base_url=_BASE_URL, api_key=api_key)

    @retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, max=30))
    def generate(
        self,
        messages: list[ChatMessage],
        *,
        temperature: float = 1.0,
        max_tokens: int = 2048,
        prefill: str | None = None,
        stop: list[str] | None = None,
        seed: int | None = None,
    ) -> GenerationResult:
        if prefill:
            raise NotImplementedError("OpenRouter backend does not support assistant prefill")
        # Disable hidden reasoning where the provider honours it.
        extra_body = {"reasoning": {"enabled": False}} if not self.thinking else {}
        resp = self._client.chat.completions.create(
            model=self.or_id,
            messages=[m.as_dict() for m in messages],
            temperature=temperature,
            max_tokens=max_tokens,
            stop=stop,
            seed=seed,
            extra_body=extra_body,
        )
        choice = resp.choices[0]
        usage = {}
        if resp.usage:
            usage = {
                "prompt_tokens": resp.usage.prompt_tokens,
                "completion_tokens": resp.usage.completion_tokens,
            }
        return GenerationResult(
            text=choice.message.content or "",
            model=self.or_id,
            finish_reason=choice.finish_reason,
            usage=usage,
        )
