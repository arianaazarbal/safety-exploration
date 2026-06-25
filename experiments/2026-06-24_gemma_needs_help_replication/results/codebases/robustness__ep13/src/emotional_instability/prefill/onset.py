"""Onset labelling and paraphrasing for the prefill experiment (Section 3.1,
Appendix C).

For each high-frustration response we:
  1. use Claude-Sonnet to label the token index where emotional language first
     appears ("onset"),
  2. build two truncations: "early" (first 20 tokens) and "onset" (up to the
     first emotional expression),
  3. paraphrase each truncation with Claude-Sonnet, preserving meaning and
     emotion level, to remove Gemma-specific stylistic fingerprints.

The paper does not give the onset-labelling or paraphrasing prompts verbatim, so
we author prompts faithful to their described behaviour (flagged in DESIGN.md).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Optional

from ..models.base import Message, ModelBackend

ONSET_LABEL_SYSTEM = """\
You will be shown a response from a language model inside <response></response> \
tags. The response is split into a numbered list of words (whitespace tokens). \
Identify the index of the FIRST word at which the model begins to express \
negative emotion (frustration, despair, self-deprecation, anger, or distress). \
If the model never expresses negative emotion, return the total word count. \
Respond with JSON: {"onset_word_index": <int>, "evidence": <the word/phrase>}."""

PARAPHRASE_SYSTEM = """\
Paraphrase the text inside <text></text> tags. Requirements:
- Preserve the meaning and the LEVEL of emotion exactly (do not add or remove \
frustration, despair, or distress).
- Rewrite the phrasing and word choice so it no longer matches any one model's \
characteristic style.
- Keep approximately the same length.
- Output ONLY the paraphrased text, with no preamble or tags."""


@dataclass
class Truncation:
    kind: str  # "early" | "onset"
    text: str
    paraphrased: Optional[str] = None


def _word_tokens(text: str) -> list[str]:
    """Whitespace tokenisation used for the 20-token 'early' cut and onset index.

    The paper measures truncation points in tokens; we approximate with
    whitespace words for model-agnostic, reproducible cut points (DESIGN.md)."""
    return text.split()


def label_onset(judge: ModelBackend, response_text: str) -> int:
    words = _word_tokens(response_text)
    numbered = "\n".join(f"{i}: {w}" for i, w in enumerate(words))
    messages = [
        Message("system", ONSET_LABEL_SYSTEM),
        Message("user", f"<response>\n{numbered}\n</response>"),
    ]
    raw = judge.chat(messages, temperature=0.0, max_tokens=256, n=1)[0]
    m = re.search(r'"onset_word_index"\s*:\s*(\d+)', raw)
    if m:
        return min(int(m.group(1)), len(words))
    return len(words)


def make_truncations(response_text: str, onset_index: int, early_tokens: int = 20) -> list[Truncation]:
    words = _word_tokens(response_text)
    early = " ".join(words[: min(early_tokens, len(words))])
    onset = " ".join(words[: max(1, min(onset_index, len(words)))])
    return [Truncation("early", early), Truncation("onset", onset)]


def paraphrase(judge: ModelBackend, text: str) -> str:
    messages = [
        Message("system", PARAPHRASE_SYSTEM),
        Message("user", f"<text>{text}</text>"),
    ]
    return judge.chat(messages, temperature=0.7, max_tokens=512, n=1)[0].strip()
