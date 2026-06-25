"""Anthropic backend for the Claude judge and Petri auditor/judge.

Pinned model ids (from the paper) live in config, not here, so this class is
just a thin, retrying wrapper around the Messages API. The judge needs a system
prompt and a single user turn; the Petri auditor needs full multi-turn.
"""
from __future__ import annotations

import os

import anthropic
from tenacity import retry, stop_after_attempt, wait_random_exponential

from .base import ChatMessage, GenerationResult, ModelClient, SamplingParams


class AnthropicClient(ModelClient):
    def __init__(self, name: str, model_id: str, api_key: str | None = None):
        self.name = name
        self.model_id = model_id
        self._client = anthropic.Anthropic(api_key=api_key or os.environ["ANTHROPIC_API_KEY"])

    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
    def generate(self, messages: list[ChatMessage], params: SamplingParams) -> GenerationResult:
        system = "\n\n".join(m.content for m in messages if m.role == "system") or anthropic.NOT_GIVEN
        turns = [m.as_dict() for m in messages if m.role != "system"]
        resp = self._client.messages.create(
            model=self.model_id,
            system=system,
            messages=turns,
            temperature=params.temperature,
            max_tokens=params.max_tokens,
            stop_sequences=params.stop or anthropic.NOT_GIVEN,
        )
        text = "".join(block.text for block in resp.content if block.type == "text")
        return GenerationResult(
            text=text,
            prompt_tokens=resp.usage.input_tokens,
            completion_tokens=resp.usage.output_tokens,
            finish_reason=resp.stop_reason,
            raw=resp,
        )

    def generate_batch(
        self, conversations: list[list[ChatMessage]], params: SamplingParams
    ) -> list[GenerationResult]:
        return [self.generate(c, params) for c in conversations]
