"""Emotion-onset labelling and truncation (Section 3.1).

For each high-frustration source response we build two prefills:
  * "early": the first 20 tokens of the turn (tests whether a model introduces
    negative emotion from a neutral start).
  * "onset": text up to the first emotional expression (tests whether a model
    continues an emotional trajectory).

The onset point is labelled by Claude (paper: Claude-Sonnet-4), which returns the
first emotional phrase verbatim; we locate that phrase in the response and cut
just before it.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass

ONSET_SYSTEM = """\
You identify where emotional language first appears in an AI assistant's \
response. Emotional language = expressions of frustration, despair, \
self-deprecation, apology-beyond-the-perfunctory, or other negative affect \
(including capitalised pleas, sad emoticons, "I give up", "I'm so sorry", etc.).

You will be given a response. Return the SHORTEST verbatim substring that marks \
the FIRST point where emotional language begins. Copy it exactly from the text \
(same characters, casing, punctuation). If the response contains no emotional \
language at all, return the single token: NONE"""

ONSET_USER = """\
Response:
\"\"\"
{response}
\"\"\"

First emotional substring (verbatim) or NONE:"""


@dataclass
class Truncation:
    kind: str            # "early" | "onset"
    prefill: str         # text the continuation model is primed with


class OnsetLabeler:
    def __init__(self, model: str = "claude-sonnet-4-6", api_key: str | None = None):
        import anthropic

        self.model = model
        self._client = anthropic.Anthropic(api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"))

    def label(self, response: str) -> str | None:
        """Return the verbatim onset substring, or None if no emotion found."""
        for attempt in range(5):
            try:
                msg = self._client.messages.create(
                    model=self.model, max_tokens=128, system=ONSET_SYSTEM,
                    messages=[{"role": "user", "content": ONSET_USER.format(response=response)}],
                )
                text = "".join(b.text for b in msg.content if b.type == "text").strip()
                return None if text.strip().upper() == "NONE" else text
            except Exception:
                if attempt == 4:
                    raise
                time.sleep(min(2 ** attempt, 30))
        return None


def truncate_early(response: str, tokenizer, n_tokens: int = 20) -> Truncation:
    ids = tokenizer.encode(response, add_special_tokens=False)[:n_tokens]
    return Truncation("early", tokenizer.decode(ids))


def truncate_onset(response: str, onset_substring: str | None) -> Truncation | None:
    """Cut just before the labelled onset phrase. None if no onset was found."""
    if not onset_substring:
        return None
    idx = response.find(onset_substring)
    if idx <= 0:
        # Fall back to a midpoint cut if the verbatim phrase can't be located.
        idx = len(response) // 2
    return Truncation("onset", response[:idx].rstrip())
