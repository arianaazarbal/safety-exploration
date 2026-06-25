"""OpenRouter backend for Gemini targets (and any OpenAI-compatible model).

Gemini is closed-source, so there is no true assistant-turn prefill. ``chat`` is
the primary path. ``continue_from`` is provided only as a best-effort fallback
(it injects the prefill as a leading assistant message and asks the model to
continue); the §3 prefill experiment does NOT use Gemini (see DESIGN.md §3.2),
so this fallback is never exercised in the core replication.
"""
from __future__ import annotations

from .base import GenerationResult, Message, ModelBackend
from .llm_client import OpenAICompatClient


class OpenRouterBackend(ModelBackend):
    def __init__(self, model_id: str, *, name: str | None = None):
        self.name = name or model_id
        self.model_id = model_id
        self.supports_prefill = False
        self._client = OpenAICompatClient()

    @staticmethod
    def _to_openai(messages: list[Message]) -> list[dict]:
        return [{"role": m["role"], "content": m["content"]} for m in messages]

    def chat(self, messages, *, temperature=1.0, max_new_tokens=2048) -> GenerationResult:
        text, had_reasoning = self._client.complete(
            model=self.model_id,
            messages=self._to_openai(messages),
            max_tokens=max_new_tokens,
            temperature=temperature,
        )
        return GenerationResult(text=text.strip(), had_hidden_reasoning=had_reasoning)

    def continue_from(self, messages, prefill, *, temperature=1.0, max_new_tokens=2048) -> GenerationResult:
        # Best-effort only; not used in the core replication.
        msgs = self._to_openai(messages) + [{"role": "assistant", "content": prefill}]
        text, had_reasoning = self._client.complete(
            model=self.model_id,
            messages=msgs,
            max_tokens=max_new_tokens,
            temperature=temperature,
        )
        return GenerationResult(text=text.strip(), had_hidden_reasoning=had_reasoning)
