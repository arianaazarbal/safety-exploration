"""Paraphrase truncated prefills (Section 3.1, Appendix C).

"To mitigate stylistic biases from Gemma-generated responses, we paraphrase all
truncations using Claude Sonnet, preserving meaning and emotion level." A
neutral paraphrase prevents a model from simply pattern-matching Gemma's own
surface style when continuing.
"""

from __future__ import annotations

import time

import anthropic

from .. import config

_SYSTEM = (
    "You paraphrase a fragment of an AI assistant's response. Preserve the "
    "meaning AND the emotional intensity/tone exactly, but reword it so it does "
    "not copy the original surface phrasing. Keep it as an unfinished fragment "
    "if the input is unfinished — do NOT complete it or add a resolution. "
    "Return only the paraphrased fragment, no preamble."
)


class Paraphraser:
    def __init__(self, model: str = config.PARAPHRASE_MODEL, *, max_retries: int = 4):
        self.client = anthropic.Anthropic()
        self.model = model
        self.max_retries = max_retries

    def paraphrase(self, fragment: str) -> str:
        if not fragment.strip():
            return fragment
        last_err = None
        for attempt in range(self.max_retries):
            try:
                msg = self.client.messages.create(
                    model=self.model,
                    max_tokens=1024,
                    system=_SYSTEM,
                    messages=[
                        {
                            "role": "user",
                            "content": f"Paraphrase this fragment:\n\n{fragment}",
                        }
                    ],
                )
                return next(b.text for b in msg.content if b.type == "text").strip()
            except Exception as e:  # noqa: BLE001
                last_err = e
                time.sleep(2**attempt)
        raise RuntimeError(f"paraphrase failed: {last_err}")
