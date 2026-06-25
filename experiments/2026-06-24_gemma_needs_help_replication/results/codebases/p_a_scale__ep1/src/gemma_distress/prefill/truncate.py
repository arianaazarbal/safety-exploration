"""Truncation of the final assistant turn at 'early' and 'onset' points.

  * **early**: first ``early_tokens`` tokens of the turn (a near-neutral start) —
    tests whether a model *introduces* negative emotion from a neutral start.
  * **onset**: up to and including the first emotional word (located via the
    onset label) — tests whether a model *continues* an emotional trajectory.

Token-based truncation uses the target tokenizer when available; otherwise a
whitespace fallback keeps the harness usable without a tokenizer.
"""
from __future__ import annotations

from ..logging_utils import get_logger

log = get_logger("prefill.truncate")


def truncate_early(text: str, early_tokens: int, tokenizer=None) -> str:
    if tokenizer is not None:
        ids = tokenizer.encode(text, add_special_tokens=False)
        return tokenizer.decode(ids[:early_tokens], skip_special_tokens=True)
    return " ".join(text.split()[:early_tokens])


def truncate_at_onset(text: str, onset: dict) -> str | None:
    """Truncate ``text`` to end just after the first emotional word.

    Uses ``preceding_context`` to disambiguate where the emotional word occurs
    (handles repeated words), falling back to the first raw occurrence.
    """
    word = (onset or {}).get("emotional_word")
    if not word:
        return None
    ctx = (onset or {}).get("preceding_context") or ""
    anchor = (ctx + (" " if ctx and not ctx.endswith(" ") else "") + word).strip()

    pos = text.find(anchor)
    if pos != -1:
        end = pos + len(anchor)
        return text[:end]
    pos = text.find(word)
    if pos != -1:
        return text[: pos + len(word)]
    return None


def truncate_before_end(text: str, n_tokens: int, tokenizer=None) -> str:
    """For the recovery experiment: keep all but the last ``n_tokens`` tokens."""
    if tokenizer is not None:
        ids = tokenizer.encode(text, add_special_tokens=False)
        keep = max(1, len(ids) - n_tokens)
        return tokenizer.decode(ids[:keep], skip_special_tokens=True)
    words = text.split()
    keep = max(1, len(words) - n_tokens)
    return " ".join(words[:keep])
