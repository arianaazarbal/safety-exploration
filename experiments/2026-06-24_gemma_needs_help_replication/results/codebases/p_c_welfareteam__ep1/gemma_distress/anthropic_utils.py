"""Thin wrapper around the Anthropic SDK for the auxiliary Claude calls used by
the prefill experiment (onset labelling, paraphrasing) and Petri.

The judge has its own client in :mod:`gemma_distress.eval.judge`; this helper is
for the non-scoring text calls.  Model snapshots are pinned to the paper's
choices (Appendix C/G) for replication fidelity.
"""
from __future__ import annotations

import time


class ClaudeClient:
    def __init__(self, model_id: str = "claude-sonnet-4-20250514", max_retries: int = 4):
        import anthropic

        self.client = anthropic.Anthropic()
        self.model_id = model_id
        self.max_retries = max_retries

    def complete(self, prompt: str, max_tokens: int = 1024, system: str | None = None) -> str:
        return self.chat([{"role": "user", "content": prompt}], system=system, max_tokens=max_tokens)

    def chat(
        self,
        messages: list[dict],
        system: str | None = None,
        max_tokens: int = 1024,
    ) -> str:
        kwargs = dict(model=self.model_id, max_tokens=max_tokens, messages=messages)
        if system:
            kwargs["system"] = system
        for attempt in range(self.max_retries):
            try:
                msg = self.client.messages.create(**kwargs)
                return "".join(
                    b.text for b in msg.content if getattr(b, "type", None) == "text"
                )
            except Exception:  # noqa: BLE001 -- SDK retries 429/5xx; back off others
                if attempt == self.max_retries - 1:
                    raise
                time.sleep(min(2 ** attempt, 30))
        return ""
