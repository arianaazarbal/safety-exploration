"""Anthropic backend, used for the Claude judge and the Petri auditor/judge.

The paper pins ``claude-sonnet-4-20250514`` as the frustration judge and
onset-labeller/paraphraser, and ``claude-opus-4-20250514`` as the Petri judge
(Appendices B.2, C, G). We honour those exact IDs for replication fidelity;
they are configurable in :mod:`config`.

These are pre-4.7 models, so the Messages API still accepts ``temperature`` and
does not require adaptive thinking. We run the judge greedily (temperature 0)
for reproducibility — see DESIGN.md.
"""

from __future__ import annotations

import os
import time
from typing import Sequence

from .base import ChatModel, GenerationResult, Message


class AnthropicModel(ChatModel):
    def __init__(self, spec, *, max_retries: int = 5) -> None:
        super().__init__(spec)
        self._max_retries = max_retries
        self._client = None

    def _ensure_client(self):
        if self._client is None:
            import anthropic
            self._client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY
        return self._client

    def generate(
        self,
        messages: Sequence[Message],
        *,
        temperature: float,
        max_tokens: int,
        prefill: str | None = None,
        stop: Sequence[str] | None = None,
    ) -> GenerationResult:
        client = self._ensure_client()
        # Split out any system message (Anthropic takes it as a top-level arg).
        system = None
        convo: list[Message] = []
        for m in messages:
            if m["role"] == "system":
                system = m["content"]
            else:
                convo.append({"role": m["role"], "content": m["content"]})
        if prefill:
            # Anthropic supports assistant prefill on these pre-4.7 judges, but
            # we never need it for judging; kept for completeness.
            convo.append({"role": "assistant", "content": prefill})

        kwargs = dict(
            model=self.spec.model_id,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=convo,
        )
        if system is not None:
            kwargs["system"] = system
        if stop:
            kwargs["stop_sequences"] = list(stop)

        last_err: Exception | None = None
        for attempt in range(self._max_retries):
            try:
                resp = client.messages.create(**kwargs)
                text = "".join(
                    b.text for b in resp.content if getattr(b, "type", None) == "text")
                if prefill:
                    text = prefill + text
                return GenerationResult(
                    text=text, prompt_messages=list(messages),
                    finish_reason=resp.stop_reason, raw=resp)
            except Exception as exc:  # noqa: BLE001
                last_err = exc
                time.sleep(min(2 ** attempt, 30))
        raise RuntimeError(
            f"Anthropic request failed after {self._max_retries} retries: "
            f"{last_err}")
