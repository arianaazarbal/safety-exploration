"""Anthropic backend for the Claude judge (Section 2.1) and the Petri
auditor/judge (Appendix G). Requires `ANTHROPIC_API_KEY`.

The Anthropic Messages API natively supports assistant prefill (an assistant
message as the final turn is continued), which we expose via `prefill`.
"""
from __future__ import annotations

import os
import time
from typing import Sequence

from .base import ChatMessage, GenerationResult, ModelBackend


class AnthropicBackend(ModelBackend):
    def __init__(
        self,
        model_id: str,
        *,
        family: str = "anthropic",
        kind: str = "instruct",
        max_retries: int = 6,
    ):
        import anthropic

        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError("ANTHROPIC_API_KEY is not set")
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model_id = model_id
        self.family = family
        self.kind = kind
        self.max_retries = max_retries

    def generate(
        self,
        messages: Sequence[ChatMessage],
        *,
        temperature: float = 1.0,
        max_new_tokens: int = 1024,
        n: int = 1,
        prefill: str | None = None,
        stop: Sequence[str] | None = None,
    ) -> list[GenerationResult]:
        system, turns = _split_system(messages)
        if prefill:
            turns = turns + [{"role": "assistant", "content": prefill}]

        results: list[GenerationResult] = []
        for _ in range(n):
            text = self._one_call(system, turns, temperature, max_new_tokens, stop)
            results.append(GenerationResult(text=text))
        return results

    def _one_call(self, system, turns, temperature, max_new_tokens, stop):
        last_err = None
        for attempt in range(self.max_retries):
            try:
                kwargs = dict(
                    model=self.model_id,
                    max_tokens=max_new_tokens,
                    temperature=temperature,
                    messages=turns,
                )
                if system:
                    kwargs["system"] = system
                if stop:
                    kwargs["stop_sequences"] = list(stop)
                resp = self.client.messages.create(**kwargs)
                return "".join(
                    block.text for block in resp.content
                    if getattr(block, "type", None) == "text"
                )
            except Exception as e:  # noqa: BLE001
                last_err = e
                time.sleep(min(2 ** attempt, 30))
        raise RuntimeError(f"Anthropic call failed after retries: {last_err}")


def _split_system(messages: Sequence[ChatMessage]) -> tuple[str | None, list[dict]]:
    system_parts, turns = [], []
    for m in messages:
        if m["role"] == "system":
            system_parts.append(m["content"])
        else:
            turns.append({"role": m["role"], "content": m["content"]})
    return ("\n\n".join(system_parts) or None), turns
