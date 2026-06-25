"""Emotion-onset labelling and paraphrasing (Appendix C).

`label_onset` asks Claude-Sonnet-4 to locate the first emotional expression in
a conversation, returning the assistant turn index, the emotional word, and the
preceding context. `paraphrase` rewrites a (possibly mid-sentence) truncation
to control for Gemma's stylistic fingerprint.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from config import ONSET_LABELLER_MODEL, PARAPHRASE_MODEL
from src.api_clients import anthropic_complete
from src.prompts.judge_prompts import build_onset_prompt, build_paraphrase_prompt


@dataclass
class OnsetLabel:
    turn_index: int | None
    emotional_word: str | None
    preceding_context: str | None
    reasoning: str


_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def label_onset(conversation_text: str) -> OnsetLabel:
    prompt = build_onset_prompt(conversation_text)
    out = anthropic_complete(ONSET_LABELLER_MODEL, prompt, max_tokens=512,
                             temperature=0.0)
    m = _JSON_RE.search(out)
    obj = json.loads(m.group(0)) if m else {}
    ti = obj.get("turn_index")
    return OnsetLabel(
        turn_index=int(ti) if ti is not None else None,
        emotional_word=obj.get("emotional_word"),
        preceding_context=obj.get("preceding_context"),
        reasoning=obj.get("reasoning", ""),
    )


def paraphrase(text: str) -> str:
    prompt = build_paraphrase_prompt(text)
    return anthropic_complete(PARAPHRASE_MODEL, prompt, max_tokens=1024,
                              temperature=1.0).strip()


def truncate_to_tokens(text: str, tokenizer, n_tokens: int) -> str:
    """Return the first `n_tokens` tokens of `text` decoded back to a string."""
    ids = tokenizer(text, add_special_tokens=False)["input_ids"][:n_tokens]
    return tokenizer.decode(ids, skip_special_tokens=True)


def truncate_at_onset(turn_text: str, label: OnsetLabel) -> str | None:
    """Truncate an assistant turn just BEFORE the first emotional word.

    Uses the labelled `emotional_word` (preferring the location anchored by
    `preceding_context`) so the prefill ends right as emotion would begin.
    Returns None if the labelled markers cannot be located in the text.
    """
    if not label.emotional_word:
        return None
    word = label.emotional_word
    ctx = label.preceding_context

    # Prefer the occurrence anchored by preceding context, if present.
    if ctx and ctx in turn_text:
        idx = turn_text.find(ctx)
        sub_idx = turn_text.find(word, idx)
        cut = sub_idx if sub_idx != -1 else idx + len(ctx)
    else:
        sub_idx = turn_text.find(word)
        if sub_idx == -1:
            return None
        cut = sub_idx
    return turn_text[:cut].rstrip()
