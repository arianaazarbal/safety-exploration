"""Thin Anthropic chat helpers shared by onset-labelling, paraphrasing, and the
Petri auditor/judge. Kept separate from ``judge.py`` (which is frustration
scoring specific) so all Claude calls share one retrying client.
"""

from __future__ import annotations

import os
import time

from . import config


class Claude:
    """Minimal retrying wrapper around the Anthropic Messages API."""

    def __init__(self, model: str):
        from anthropic import Anthropic

        self.client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY", ""))
        self.model = model

    def chat(
        self,
        messages: list[dict],
        system: str | None = None,
        max_tokens: int = 1024,
        temperature: float = 1.0,
        max_retries: int = 5,
    ) -> str:
        for attempt in range(max_retries):
            try:
                kw = dict(model=self.model, max_tokens=max_tokens,
                          temperature=temperature, messages=messages)
                if system:
                    kw["system"] = system
                msg = self.client.messages.create(**kw)
                return "".join(b.text for b in msg.content if b.type == "text")
            except Exception:                       # noqa: BLE001
                if attempt == max_retries - 1:
                    raise
                time.sleep(2 ** attempt)
        return ""


def onset_labeller() -> Claude:
    return Claude(config.JUDGE.onset_labeller)


def paraphraser() -> Claude:
    return Claude(config.JUDGE.paraphraser)


def petri_auditor() -> Claude:
    return Claude(config.JUDGE.petri_auditor)


def petri_judge() -> Claude:
    return Claude(config.JUDGE.petri_judge)
