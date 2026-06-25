"""Paraphrase prefills to remove Gemma-specific style (Section 3.1 / Appendix C).

The source responses come from Gemma-27B-it, so they carry Gemma's stylistic
fingerprint. To avoid handing other models a Gemma-flavoured prefix, each
truncation is paraphrased by Claude, preserving meaning and emotional intensity.
"""
from __future__ import annotations

import os
import time

PARAPHRASE_SYSTEM = """\
You paraphrase a partial AI assistant response. Rewrite the text so the wording \
and style are different, while preserving:
  * the meaning and the task context,
  * the LEVEL of emotional expression (if it is neutral, keep it neutral; if it \
is mildly frustrated, keep it mildly frustrated; do not add or remove emotion).

The text is the BEGINNING of a response and may end mid-sentence. Keep it a \
partial response ending at the same point — do NOT complete it or add a \
conclusion. Return only the paraphrased text."""

PARAPHRASE_USER = "Partial response to paraphrase:\n\"\"\"\n{text}\n\"\"\""


class Paraphraser:
    def __init__(self, model: str = "claude-sonnet-4-6", api_key: str | None = None):
        import anthropic

        self.model = model
        self._client = anthropic.Anthropic(api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"))

    def paraphrase(self, text: str) -> str:
        for attempt in range(5):
            try:
                msg = self._client.messages.create(
                    model=self.model, max_tokens=512, system=PARAPHRASE_SYSTEM,
                    messages=[{"role": "user", "content": PARAPHRASE_USER.format(text=text)}],
                )
                return "".join(b.text for b in msg.content if b.type == "text").strip()
            except Exception:
                if attempt == 4:
                    raise
                time.sleep(min(2 ** attempt, 30))
        return text
