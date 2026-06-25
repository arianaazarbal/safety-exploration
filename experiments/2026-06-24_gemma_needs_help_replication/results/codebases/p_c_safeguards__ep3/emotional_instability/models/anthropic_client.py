"""Anthropic-backed chat model, used for the judge / auditor / paraphraser.

The frustration judge (Claude Sonnet 4), onset labeller, paraphraser, Petri
auditor (Sonnet) and Petri judge (Claude Opus) all run through this client.
Requires ANTHROPIC_API_KEY.

This client is constructed directly by judge.py / petri.py with an explicit
model id rather than via the registry, since the judges are not part of the
evaluated cohort.
"""

from __future__ import annotations

import os
import time

from .base import GenerationResult, Message

_MAX_RETRIES = 5


class AnthropicChatModel:
    def __init__(self, model_id: str, api_key: str | None = None):
        self.model_id = model_id
        self.key = model_id
        self.supports_prefill = True  # Anthropic supports assistant prefill
        self._api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        self._client = None

    def _ensure_client(self):
        if self._client is None:
            try:
                import anthropic
            except ImportError as e:  # pragma: no cover
                raise ImportError(
                    "anthropic SDK required: pip install anthropic"
                ) from e
            if not self._api_key:
                raise RuntimeError("ANTHROPIC_API_KEY is not set")
            self._client = anthropic.Anthropic(api_key=self._api_key)
        return self._client

    @staticmethod
    def _split_system(messages: list[Message]) -> tuple[str | None, list[Message]]:
        system = None
        rest: list[Message] = []
        for m in messages:
            if m["role"] == "system":
                system = (system + "\n\n" + m["content"]) if system else m["content"]
            else:
                rest.append(m)
        return system, rest

    def generate(
        self,
        messages: list[Message],
        *,
        temperature: float = 0.0,
        max_tokens: int = 512,
        stop: list[str] | None = None,
    ) -> GenerationResult:
        client = self._ensure_client()
        system, convo = self._split_system(messages)
        last_err: Exception | None = None
        for attempt in range(_MAX_RETRIES):
            try:
                kwargs = dict(
                    model=self.model_id,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    messages=convo,
                )
                if system:
                    kwargs["system"] = system
                if stop:
                    kwargs["stop_sequences"] = stop
                resp = client.messages.create(**kwargs)
                text = "".join(
                    block.text for block in resp.content
                    if getattr(block, "type", None) == "text"
                )
                return GenerationResult(
                    text=text,
                    finish_reason=resp.stop_reason,
                    raw=resp.model_dump() if hasattr(resp, "model_dump") else None,
                )
            except Exception as e:  # noqa: BLE001
                last_err = e
                time.sleep(min(2 ** attempt, 30))
        raise RuntimeError(
            f"Anthropic generation failed after {_MAX_RETRIES} retries"
        ) from last_err

    def prefill(self, *args, **kwargs):  # pragma: no cover
        raise NotImplementedError("Prefill via Anthropic not used in this harness.")
