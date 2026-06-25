"""Token-aware truncation helpers for the prefill experiment.

The paper truncates "20 tokens into the turn" and at the emotional-onset token,
using the model's own tokenizer. We use the Gemma instruct tokenizer when
available and fall back to whitespace tokens otherwise (so the pipeline still
runs without the heavy stack installed).
"""
from __future__ import annotations

from functools import lru_cache


@lru_cache(maxsize=4)
def _hf_tokenizer(model_id: str):
    try:
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained(model_id)
    except Exception:
        return None


def truncate_to_tokens(text: str, n_tokens: int, model_id: str = "google/gemma-3-27b-it") -> str:
    """Return ``text`` truncated to its first ``n_tokens`` tokens."""
    tok = _hf_tokenizer(model_id)
    if tok is not None:
        ids = tok(text, add_special_tokens=False)["input_ids"][:n_tokens]
        return tok.decode(ids, skip_special_tokens=True)
    return " ".join(text.split()[:n_tokens])


def truncate_before_substring(text: str, marker: str, preceding_context: str = "") -> str:
    """Cut ``text`` just before ``marker`` (the emotional word).

    If ``preceding_context`` is supplied, prefer the occurrence of ``marker``
    that immediately follows it (the onset labeller guarantees this pairing),
    which disambiguates repeated words.
    """
    if preceding_context:
        anchor = preceding_context.rstrip()
        idx = text.find(anchor)
        if idx != -1:
            after = text.find(marker, idx)
            if after != -1:
                return text[:after].rstrip()
    idx = text.find(marker)
    if idx != -1:
        return text[:idx].rstrip()
    return text  # marker not found: keep whole turn
