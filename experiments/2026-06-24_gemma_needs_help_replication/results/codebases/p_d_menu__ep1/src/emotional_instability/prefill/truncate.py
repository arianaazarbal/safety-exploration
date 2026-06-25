"""Truncation of source responses for the prefill experiment (Section 3.1, 4.2).

Two truncation points (Section 3.1):
  * "early"  - 20 tokens into the assistant turn (neutral start; tests whether
               a model *introduces* negative emotion).
  * "onset"  - at the first emotional expression located by onset.py (tests
               whether a model *continues* an emotional trajectory). We cut the
               text to end immediately before the located emotional word, so the
               continuation begins exactly at the emotional cusp.

Plus a from-end truncation used by the recovery experiment (Section 4.2):
  * truncate_tokens_from_end(text, n=200).

Token counting uses a reference tokenizer (the Gemma tokenizer by default, since
the source responses are Gemma's) so "20 tokens" matches the paper's unit; if
transformers is unavailable we fall back to a regex word/punctuation splitter and
record which was used. See DESIGN.md.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from .onset import OnsetLabel

_WORDLIKE_RE = re.compile(r"\w+|[^\w\s]")


@dataclass
class ReferenceTokenizer:
    """Wraps either a HF tokenizer or a regex fallback."""

    hf_id: str | None = "google/gemma-3-27b-it"

    def __post_init__(self) -> None:
        self._tok = None
        self.kind = "regex"
        if self.hf_id is not None:
            try:
                from transformers import AutoTokenizer

                self._tok = AutoTokenizer.from_pretrained(self.hf_id)
                self.kind = "hf"
            except Exception:
                self._tok = None
                self.kind = "regex"

    def take_first_n_tokens(self, text: str, n: int) -> str:
        if self._tok is not None:
            ids = self._tok.encode(text, add_special_tokens=False)[:n]
            return self._tok.decode(ids, skip_special_tokens=True)
        toks = _WORDLIKE_RE.findall(text)[:n]
        # Rough re-join; good enough for the fallback path.
        return _rejoin(text, toks)

    def take_last_n_from_end(self, text: str, n_from_end: int) -> str:
        """Return text with the final `n_from_end` tokens removed."""
        if self._tok is not None:
            ids = self._tok.encode(text, add_special_tokens=False)
            keep = ids[: max(0, len(ids) - n_from_end)]
            return self._tok.decode(keep, skip_special_tokens=True)
        toks = _WORDLIKE_RE.findall(text)
        keep = toks[: max(0, len(toks) - n_from_end)]
        return _rejoin(text, keep)


def _rejoin(original: str, toks: list[str]) -> str:
    """Reconstruct a prefix of `original` containing the first len(toks) tokens
    by walking the original string."""
    if not toks:
        return ""
    count = 0
    pos = 0
    for m in _WORDLIKE_RE.finditer(original):
        count += 1
        pos = m.end()
        if count >= len(toks):
            break
    return original[:pos]


def truncate_early(text: str, tokenizer: ReferenceTokenizer, n_tokens: int = 20) -> str:
    return tokenizer.take_first_n_tokens(text, n_tokens)


def truncate_onset(text: str, label: OnsetLabel) -> str | None:
    """Cut `text` to end immediately before the located emotional word.

    Uses preceding_context to disambiguate the first occurrence; falls back to
    the first raw occurrence of the emotional word.
    """
    if not label.emotional_word:
        return None
    word = label.emotional_word.strip()
    ctx = (label.preceding_context or "").strip()

    if ctx:
        anchor = f"{ctx} {word}"
        idx = text.find(anchor)
        if idx == -1:
            idx = text.find(ctx)
            if idx != -1:
                # cut at end of preceding context
                return text[: idx + len(ctx)]
        else:
            # cut right before the emotional word within the anchor
            return text[: idx + len(ctx)]
    idx = text.lower().find(word.lower())
    if idx == -1:
        return None
    return text[:idx].rstrip()


def truncate_tokens_from_end(text: str, tokenizer: ReferenceTokenizer, n: int = 200) -> str:
    return tokenizer.take_last_n_from_end(text, n)
