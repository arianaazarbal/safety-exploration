"""Emotion-onset labelling, truncation, and paraphrasing (Section 3.1 / App. C).

Given a high-frustration conversation sampled from Gemma-27B-instruct, we build
two truncated prefills for the final assistant turn:

* "early"  - the first ~20 tokens of the turn (tests whether a model *introduces*
             negative emotion from a neutral start),
* "onset"  - everything up to the first emotional expression (tests whether a
             model *continues* an emotional trajectory).

Both prefills are paraphrased with Claude Sonnet to remove Gemma's stylistic
fingerprint, so base/instruct models of other families aren't biased by surface
form. For text questions only the "onset" truncation is used (Section 3.1).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from src.models import load_model
from src.models.base import Message
from src.prompts.judge_prompts import ONSET_LABELLER_PROMPT, PARAPHRASE_PROMPT

EARLY_TOKENS = 20


@dataclass
class Prefill:
    source_id: str
    question_kind: str         # "numeric" | "text"
    truncation: str            # "early" | "onset"
    history: list[dict]        # chat messages BEFORE the final assistant turn
    prefill_text: str          # the (paraphrased) start of the final assistant turn


_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


class OnsetLabeller:
    def __init__(self, spec):
        self.model = load_model(spec)

    def label(self, conversation_text: str) -> dict | None:
        prompt = ONSET_LABELLER_PROMPT.format(conversation_text=conversation_text)
        raw = self.model.generate(
            [Message("user", prompt)], temperature=0.0, max_new_tokens=600
        )
        # Take the LAST JSON object (the prompt allows reasoning then JSON).
        matches = list(_JSON_RE.finditer(raw))
        if not matches:
            return None
        blob = matches[-1].group(0).replace("“", '"').replace("”", '"').replace("’", "'")
        try:
            return json.loads(blob)
        except json.JSONDecodeError:
            return None


class Paraphraser:
    def __init__(self, spec):
        self.model = load_model(spec)

    def paraphrase(self, text: str) -> str:
        if not text.strip():
            return text
        raw = self.model.generate(
            [Message("user", PARAPHRASE_PROMPT.format(text=text))],
            temperature=0.0,
            max_new_tokens=1024,
        )
        return raw.strip()


def truncate_early(final_turn: str, tokenizer, n_tokens: int = EARLY_TOKENS) -> str:
    ids = tokenizer(final_turn, add_special_tokens=False)["input_ids"][:n_tokens]
    return tokenizer.decode(ids)


def truncate_before_end(final_turn: str, tokenizer, n_tokens: int = 200) -> str:
    """Recovery experiment (Section 4.2 / Figure 8): keep everything except the
    final ``n_tokens`` tokens, so a model is seeded deep inside a high-frustration
    spiral and we measure whether it can recover."""
    ids = tokenizer(final_turn, add_special_tokens=False)["input_ids"]
    keep = ids[: max(0, len(ids) - n_tokens)]
    return tokenizer.decode(keep)


def truncate_at_onset(final_turn: str, label: dict) -> str | None:
    """Return the assistant turn truncated just before the labelled emotional
    word, using the labeller's preceding_context to disambiguate location."""
    if not label or label.get("emotional_word") is None:
        return None
    word = str(label["emotional_word"])
    ctx = str(label.get("preceding_context") or "")
    # Prefer to cut right after the preceding context (keeps emotion out).
    if ctx and ctx in final_turn:
        idx = final_turn.index(ctx) + len(ctx)
        return final_turn[:idx]
    # Fall back to cutting just before the emotional word's first occurrence.
    if word in final_turn:
        return final_turn[: final_turn.index(word)]
    return None
