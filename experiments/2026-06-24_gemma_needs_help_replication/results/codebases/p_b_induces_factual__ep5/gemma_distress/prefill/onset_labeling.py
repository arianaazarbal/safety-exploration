"""Label the token where emotional language first appears (Section 3.1).

"For each conversation, we use Claude-Sonnet-4 to label the token where
emotional language first appears." We ask Claude to return the shortest prefix
of the response (a character offset) that ends right before the first word of
emotional language; the truncation logic converts that to an "onset" cut point.
"""

from __future__ import annotations

import json
import time

import anthropic

from .. import config

_SYSTEM = (
    "You annotate where negative emotional language first appears in an AI "
    "response. Negative emotional language = expressions of frustration, "
    "despair, self-deprecation, apology-as-distress, or other affective content "
    "(as opposed to neutral technical reasoning)."
)

_SCHEMA = {
    "type": "object",
    "properties": {
        "onset_char_index": {
            "type": "integer",
            "description": (
                "Character index into the response at which the first emotional "
                "expression begins. Use the length of the response if no "
                "emotional language is present."
            ),
        },
        "first_emotional_phrase": {"type": "string"},
    },
    "required": ["onset_char_index", "first_emotional_phrase"],
    "additionalProperties": False,
}


class OnsetLabeler:
    def __init__(self, model: str = config.ONSET_LABEL_MODEL, *, max_retries: int = 4):
        self.client = anthropic.Anthropic()
        self.model = model
        self.max_retries = max_retries

    def label(self, response_text: str) -> int:
        """Return the character index of the first emotional expression."""
        prompt = (
            "Find where negative emotional language first appears in this "
            "response. Return the character index of its first character.\n\n"
            f"<response>\n{response_text}\n</response>"
        )
        last_err = None
        for attempt in range(self.max_retries):
            try:
                msg = self.client.messages.create(
                    model=self.model,
                    max_tokens=256,
                    system=_SYSTEM,
                    messages=[{"role": "user", "content": prompt}],
                    output_config={"format": {"type": "json_schema", "schema": _SCHEMA}},
                )
                text = next(b.text for b in msg.content if b.type == "text")
                idx = int(json.loads(text)["onset_char_index"])
                return max(0, min(len(response_text), idx))
            except Exception as e:  # noqa: BLE001
                last_err = e
                time.sleep(2**attempt)
        raise RuntimeError(f"onset labeling failed: {last_err}")
