"""Truncation of seed responses at the "early" and "onset" points (Section 3.1).

* early  — 20 tokens into the assistant turn (tests whether a model introduces
           negative emotion from a near-neutral start).
* onset  — at the first emotional expression (tests whether a model continues an
           existing emotional trajectory).
"""

from __future__ import annotations

from functools import lru_cache

from ..config import PREFILL
from .onset import OnsetLabel


@lru_cache(maxsize=4)
def _tokenizer(model_id: str = "google/gemma-3-27b-it"):
    try:
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained(model_id)
    except Exception:  # noqa: BLE001 - offline: fall back to whitespace tokens
        return None


def truncate_early(text: str, n_tokens: int = PREFILL.early_truncation_tokens,
                   model_id: str = "google/gemma-3-27b-it") -> str:
    tok = _tokenizer(model_id)
    if tok is None:
        return " ".join(text.split()[:n_tokens])
    ids = tok.encode(text, add_special_tokens=False)[:n_tokens]
    return tok.decode(ids, skip_special_tokens=True)


def truncate_onset(text: str, label: OnsetLabel) -> str | None:
    """Truncate ``text`` so it ends just after the first emotional phrase.

    Locates the emotional word, preferring the position right after its preceding
    context (so repeated words don't mislocate). Returns None if it can't be found.
    """
    if not label.emotional_word:
        return None
    word = label.emotional_word.strip().strip('"\'')
    ctx = (label.preceding_context or "").strip().strip('"\'')

    lower = text.lower()
    pos = -1
    if ctx:
        c = lower.find(ctx.lower())
        if c != -1:
            pos = lower.find(word.lower(), c)
    if pos == -1:
        pos = lower.find(word.lower())
    if pos == -1:
        return None
    end = pos + len(word)
    return text[:end]


def make_truncations(text: str, label: OnsetLabel, kinds: tuple[str, ...]) -> dict[str, str]:
    """Return {kind: truncated_text} for the requested truncation kinds."""
    out: dict[str, str] = {}
    for kind in kinds:
        if kind == "early":
            out["early"] = truncate_early(text)
        elif kind == "onset":
            t = truncate_onset(text, label)
            if t is not None:
                out["onset"] = t
        else:
            raise ValueError(f"Unknown truncation kind {kind!r}")
    return out
