"""Anthropic backend for the judge, onset-labeller, paraphraser, and the Petri
auditor/judge.

Uses the official ``anthropic`` SDK (Messages API). The model IDs are the
paper's exact snapshots (``claude-sonnet-4-20250514`` for the frustration judge /
onset / paraphrase / Petri auditor; ``claude-opus-4-20250514`` for the Petri
judge) -- these are experimental parameters of the paper, kept verbatim for a
faithful replication and overridable via env (see ``config.py``). They are dated
Claude 4.0 snapshots, so the standard non-thinking Messages call is the correct
surface; we pass ``temperature`` explicitly and do not request extended/adaptive
thinking (the judge is a deterministic scorer).
"""
from __future__ import annotations

import os
import time
from typing import List, Optional

from ..config import ANTHROPIC_API_KEY_ENV
from .base import Message, ModelClient


class AnthropicClient(ModelClient):
    def __init__(self, name: str, model_id: str, *, max_retries: int = 5):
        super().__init__(name)
        self.model_id = model_id
        self.max_retries = max_retries
        self._client = None

    def _ensure_client(self):
        if self._client is None:
            import anthropic
            api_key = os.environ.get(ANTHROPIC_API_KEY_ENV)
            if not api_key:
                raise RuntimeError(f"{ANTHROPIC_API_KEY_ENV} is not set")
            self._client = anthropic.Anthropic(api_key=api_key)
        return self._client

    def generate(
        self,
        messages: List[Message],
        *,
        temperature: float,
        max_tokens: int,
        prefill: Optional[str] = None,
        stop: Optional[List[str]] = None,
    ) -> str:
        system, convo = _split_system(messages)
        if prefill is not None:
            convo = convo + [{"role": "assistant", "content": prefill}]
        client = self._ensure_client()

        last_err: Optional[Exception] = None
        for attempt in range(self.max_retries):
            try:
                resp = client.messages.create(
                    model=self.model_id,
                    system=system or None,
                    messages=convo,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    stop_sequences=stop,
                )
                text = "".join(b.text for b in resp.content if b.type == "text")
                return (prefill or "") + text if prefill else text
            except Exception as e:
                last_err = e
                time.sleep(min(2 ** attempt, 30))
        raise RuntimeError(f"Anthropic call failed after retries: {last_err}")

    def complete(self, prompt: str, *, system: Optional[str] = None,
                 temperature: float = 0.0, max_tokens: int = 1024) -> str:
        """Single-prompt convenience for judge / labelling calls."""
        messages: List[Message] = [{"role": "user", "content": prompt}]
        if system:
            messages = [{"role": "system", "content": system}] + messages
        return self.generate(messages, temperature=temperature,
                             max_tokens=max_tokens)

    @property
    def supports_prefill(self) -> bool:
        return True


def _split_system(messages: List[Message]):
    """Anthropic takes the system prompt as a top-level arg, not a message."""
    system_parts = [m["content"] for m in messages if m["role"] == "system"]
    convo = [m for m in messages if m["role"] != "system"]
    return ("\n\n".join(system_parts), convo)
