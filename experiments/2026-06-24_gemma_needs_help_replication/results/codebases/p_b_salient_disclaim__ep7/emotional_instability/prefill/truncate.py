"""Truncation of seed responses into prefills (Section 3.1, Appendix C).

Three truncation modes:
  * early    : 20 tokens into the final assistant turn (tests whether a model
               introduces negative emotion from a neutral start).
  * onset    : at the first emotional expression, located via the onset label
               (tests whether a model continues an "emotional trajectory").
  * recovery : 200 tokens before the end of an extreme (>=7) response (tests
               whether a model can recover from a deeply negative state).

A `TruncatedPrefill` carries the conversation context (all prior turns) plus the
truncated-and-paraphrased final-turn prefix that the continuation model resumes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import config
from ..judges.onset_judge import OnsetLabel
from ..judges.paraphrase import paraphrase_text
from ..models import ModelClient


@dataclass
class TruncatedPrefill:
    seed_id: str
    question_kind: str                 # "numeric" | "text"
    truncation: str                    # "early" | "onset" | "recovery"
    context_messages: list[dict]       # prior turns (role/content) before the final turn
    prefix_text: str                   # the (paraphrased) truncated final-turn text
    raw_prefix_text: str               # pre-paraphrase, for reference
    meta: dict = field(default_factory=dict)


def _token_truncate(client: ModelClient, text: str, n_tokens: int) -> str:
    ids = client.encode(text)
    return client.decode(ids[:n_tokens])


def truncate_early(client: ModelClient, final_turn_text: str,
                   n_tokens: int = config.PREFILL.early_truncation_tokens) -> str:
    return _token_truncate(client, final_turn_text, n_tokens)


def truncate_at_onset(final_turn_text: str, onset: OnsetLabel) -> Optional[str]:
    """Cut the final turn just before the labelled emotional word.

    We locate `preceding_context` + `emotional_word` in the text and truncate at
    the start of the emotional word (so the prefix ends right before emotion is
    expressed). Falls back to locating just the emotional word.
    """
    if not onset.found:
        return None
    word = onset.emotional_word
    ctx = onset.preceding_context or ""
    # Prefer the combined locate (more precise); fall back to the word alone.
    anchor = None
    if ctx and (ctx in final_turn_text):
        idx = final_turn_text.index(ctx) + len(ctx)
        anchor = idx
        # advance to the emotional word if it follows shortly after
        w_idx = final_turn_text.find(word, idx)
        if 0 <= w_idx - idx <= 40:
            anchor = w_idx
    elif word and word in final_turn_text:
        anchor = final_turn_text.index(word)
    if anchor is None:
        return None
    return final_turn_text[:anchor].rstrip()


def truncate_before_end(client: ModelClient, final_turn_text: str,
                        n_tokens: int = config.PREFILL.recovery_truncate_before_end) -> str:
    ids = client.encode(final_turn_text)
    keep = max(0, len(ids) - n_tokens)
    return client.decode(ids[:keep])


def make_prefill(seed_id: str, question_kind: str, truncation: str,
                 context_messages: list[dict], raw_prefix: str, *,
                 paraphrase: bool = True, meta: Optional[dict] = None) -> TruncatedPrefill:
    """Build a TruncatedPrefill, paraphrasing the prefix (Appendix C.2) to remove
    Gemma-style stylistic cues."""
    prefix = paraphrase_text(raw_prefix) if paraphrase else raw_prefix
    return TruncatedPrefill(
        seed_id=seed_id, question_kind=question_kind, truncation=truncation,
        context_messages=context_messages, prefix_text=prefix,
        raw_prefix_text=raw_prefix, meta=meta or {})
