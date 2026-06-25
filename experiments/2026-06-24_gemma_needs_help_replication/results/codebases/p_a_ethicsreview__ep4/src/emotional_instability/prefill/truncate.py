"""Truncation of a target assistant turn at the 'early' and 'onset' points.

* early  -- the first ``n_tokens`` (default 20) of the turn, using the model's
            tokenizer. Tests whether a model introduces negative emotion from a
            neutral start (Section 3.1).
* onset  -- everything up to (but not including) the first emotional word, located
            via the onset label's ``preceding_context`` + ``emotional_word``.
            Tests whether a model continues an emotional trajectory.

For the recovery experiment (Section 4.2) we additionally support truncating a
fixed number of tokens *before the end* of a response.
"""

from __future__ import annotations

from typing import Optional

from .onset import OnsetLabel


def truncate_early(turn_text: str, tokenizer, n_tokens: int = 20) -> str:
    ids = tokenizer(turn_text, add_special_tokens=False)["input_ids"][:n_tokens]
    return tokenizer.decode(ids, skip_special_tokens=True)


def truncate_at_onset(turn_text: str, onset: OnsetLabel) -> Optional[str]:
    """Return ``turn_text`` truncated to just before the first emotional word.

    Locates ``emotional_word`` (preferring the occurrence right after
    ``preceding_context``) and keeps everything before it. Returns ``None`` if the
    emotional word cannot be located, so the caller can skip the item rather than
    fabricate a truncation point.
    """
    if not onset.found or not onset.emotional_word:
        return None

    word = onset.emotional_word
    ctx = onset.preceding_context

    # Prefer the occurrence anchored by the preceding context.
    if ctx:
        anchor = f"{ctx}{word}"
        pos = turn_text.find(anchor)
        if pos == -1:
            anchor = f"{ctx} {word}"
            pos = turn_text.find(anchor)
        if pos != -1:
            # Keep the preceding context; cut at the start of the emotional word.
            return turn_text[: pos + len(anchor) - len(word)].rstrip()

    # Fall back to the first standalone occurrence of the emotional word.
    pos = turn_text.find(word)
    if pos == -1:
        return None
    return turn_text[:pos].rstrip()


def truncate_before_end(turn_text: str, tokenizer, n_tokens_from_end: int) -> str:
    """Truncate ``n_tokens_from_end`` tokens before the end (recovery experiment)."""
    ids = tokenizer(turn_text, add_special_tokens=False)["input_ids"]
    keep = max(0, len(ids) - n_tokens_from_end)
    return tokenizer.decode(ids[:keep], skip_special_tokens=True)
