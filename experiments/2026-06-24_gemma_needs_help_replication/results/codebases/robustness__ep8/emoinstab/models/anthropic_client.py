"""Anthropic API backend.

Used for the frustration judge (claude-sonnet-4, Appendix B.2), the prefill
onset-labeller / paraphraser (Appendix C), and the Petri auditor / judge
(Appendix G). Supports assistant prefill natively (an assistant message as the
last turn), which we expose via ``continue_prefill``.
"""
from __future__ import annotations

from typing import Sequence

from emoinstab.config import ModelSpec
from emoinstab.models._api_common import require_env, threaded_map, with_retry
from emoinstab.models.base import Conversation, ModelClient, SamplingParams


class AnthropicClient(ModelClient):
    def __init__(self, spec: ModelSpec):
        super().__init__(spec)
        import anthropic

        self._client = anthropic.Anthropic(api_key=require_env("ANTHROPIC_API_KEY"))

    def _split(self, messages: Conversation):
        system = None
        turns = []
        for m in messages:
            if m.role == "system":
                system = m.content
            else:
                turns.append({"role": m.role, "content": m.content})
        return system, turns

    @with_retry
    def _once(self, system, turns, params: SamplingParams) -> str:
        kwargs = dict(
            model=self.spec.model_id,
            max_tokens=params.max_tokens,
            temperature=params.temperature,
            messages=turns,
        )
        if system:
            kwargs["system"] = system
        # Disable extended thinking unless explicitly requested.
        resp = self._client.messages.create(**kwargs)
        return "".join(
            block.text for block in resp.content if getattr(block, "type", "") == "text"
        )

    def chat(self, messages: Conversation, params: SamplingParams | None = None) -> list[str]:
        params = params or self.default_params()
        system, turns = self._split(messages)
        return [self._once(system, turns, params) for _ in range(params.n)]

    def chat_batch(
        self, conversations: Sequence[Conversation], params: SamplingParams | None = None
    ) -> list[list[str]]:
        params = params or self.default_params()

        def run(conv):
            return self.chat(conv, params)

        return threaded_map(run, list(conversations))

    def continue_prefill(
        self, messages: Conversation, prefill: str, params: SamplingParams | None = None
    ) -> list[str]:
        params = params or self.default_params()
        system, turns = self._split(messages)
        turns = turns + [{"role": "assistant", "content": prefill}]
        outs = []
        for _ in range(params.n):
            outs.append(self._once(system, turns, params))
        return outs
