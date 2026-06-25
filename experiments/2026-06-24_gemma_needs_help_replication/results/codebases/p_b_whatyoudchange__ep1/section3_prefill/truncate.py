"""Truncate an assistant response at the two prefill points (§3.1).

  - "early": 20 tokens into the turn (tests whether models *introduce* negative
    emotion from a neutral start). Token counting uses the shared Gemma tokenizer
    so base and instruct receive the identical prefill.
  - "onset": at the first emotional expression (tests whether models *continue*
    an emotional trajectory). Located via the onset label's emotional word.
"""

from __future__ import annotations

from functools import lru_cache

from config import SECTION3, TRAIN, TARGET_MODELS


@lru_cache(maxsize=1)
def _gemma_tokenizer():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(TARGET_MODELS[TRAIN.base_model].hf_id)


def truncate_early(text: str, n_tokens: int = SECTION3.early_truncation_tokens) -> str:
    tok = _gemma_tokenizer()
    ids = tok.encode(text, add_special_tokens=False)[:n_tokens]
    return tok.decode(ids, skip_special_tokens=True)


def truncate_at_onset(text: str, onset: dict) -> str | None:
    """Truncate just before the emotional word, keeping the preceding context.

    Returns None if onset labelling found no emotion (so this seed is skipped).
    """
    word = onset.get("emotional_word")
    if not word:
        return None
    ctx = onset.get("preceding_context") or ""
    # Prefer locating the emotional word right after its preceding context.
    if ctx and ctx in text:
        start = text.index(ctx) + len(ctx)
        rel = text[start:].find(word)
        if rel != -1:
            return text[: start + rel].rstrip()
    # Fallback: first standalone occurrence of the emotional word.
    idx = text.find(word)
    if idx == -1:
        return None
    return text[:idx].rstrip()


def make_truncations(response_text: str, onset: dict, *, is_numeric: bool,
                     ) -> dict[str, str]:
    """Return {condition: prefill_text} for the conditions valid for this task
    type (numeric -> early+onset, text -> onset only)."""
    out: dict[str, str] = {}
    conditions = (SECTION3.numeric_conditions if is_numeric
                  else SECTION3.text_conditions)
    if "early" in conditions:
        early = truncate_early(response_text)
        if early.strip():
            out["early"] = early
    if "onset" in conditions:
        onset_trunc = truncate_at_onset(response_text, onset)
        if onset_trunc and onset_trunc.strip():
            out["onset"] = onset_trunc
    return out
