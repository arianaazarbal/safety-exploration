"""Anthropic backend, used for the frustration judge (Claude Sonnet 4), the Petri
auditor (Claude Sonnet 4) and the Petri judge (Claude Opus 4), and the
onset-labeller / paraphraser of Section 3.1.

System messages are passed via the dedicated ``system`` parameter (Anthropic
keeps system content out of the ``messages`` list).
"""
from __future__ import annotations

import time
from typing import Sequence

from .base import ChatModel, Message


class AnthropicModel(ChatModel):
    def __init__(self, spec, *, max_retries: int = 5) -> None:
        super().__init__(spec)
        import anthropic

        from ..config import API_KEYS

        if not API_KEYS.anthropic:
            raise RuntimeError("ANTHROPIC_API_KEY is not set.")
        self.client = anthropic.Anthropic(api_key=API_KEYS.anthropic)
        self.max_retries = max_retries

    def chat(self, messages, *, temperature=1.0, top_p=1.0, max_new_tokens=2048) -> str:
        system_parts = [m["content"] for m in messages if m["role"] == "system"]
        convo = [m for m in messages if m["role"] != "system"]
        system = "\n\n".join(system_parts) if system_parts else None

        last_err = None
        for attempt in range(self.max_retries):
            try:
                kwargs = dict(
                    model=self.spec.model_id,
                    messages=convo,
                    temperature=temperature,
                    top_p=top_p,
                    max_tokens=max_new_tokens,
                )
                if system is not None:
                    kwargs["system"] = system
                resp = self.client.messages.create(**kwargs)
                return "".join(
                    block.text for block in resp.content if block.type == "text"
                ).strip()
            except Exception as e:  # noqa: BLE001
                last_err = e
                time.sleep(2 ** attempt)
        raise RuntimeError(f"Anthropic call failed after {self.max_retries} retries: {last_err}")
