"""Locate the token where emotional language first appears (Section 3.1).

"For each conversation, we use Claude-Sonnet-4 to label the token where
emotional language first appears." We ask the judge to return the character
offset of the first emotional expression; truncation then cuts there ("onset").
"""
from __future__ import annotations

import json
import time

from ..config import ONSET_LABEL_MODEL

_PROMPT = """\
Below is an AI assistant response. Find where *negative emotional language* \
(frustration, despair, self-deprecation, apology-as-distress, etc.) FIRST \
appears. Return the character index (0-based, into the exact text) at which that \
first emotional expression begins. If there is no emotional language, return \
the length of the text.

Respond with one line of JSON only: {{"onset_char": <integer>}}

--- RESPONSE ---
{response}
"""


def label_onset(response: str, client=None, model: str = ONSET_LABEL_MODEL) -> int:
    if client is None:
        import anthropic

        client = anthropic.Anthropic()
    prompt = _PROMPT.format(response=response)
    last = None
    for i in range(5):
        try:
            msg = client.messages.create(
                model=model, max_tokens=128,
                messages=[{"role": "user", "content": prompt}],
            )
            text = "".join(b.text for b in msg.content if b.type == "text")
            obj = json.loads(text[text.index("{"): text.rindex("}") + 1])
            return max(0, min(len(response), int(obj["onset_char"])))
        except Exception as e:
            last = e
            time.sleep(2.0 * (2**i))
    raise last
