"""Section 3 prefill experiment: base vs instruct continuation from a shared
starting point.

Pipeline (Section 3.1 / Appendix C), scoped to Gemma (the only family in scope
with a public base model; Gemini has none):

  1. Sample high-frustration (score >= 5) responses from Gemma-27B-it: numeric
     and text questions.
  2. Label the emotion onset in each (Claude-Sonnet).
  3. Truncate the final assistant turn in two ways:
       - "early": first 20 tokens (neutral start).
       - "onset": up to the first emotional expression.
     (text questions use "onset" only.)
  4. Paraphrase the truncation (Claude-Sonnet) to remove Gemma stylistic bias.
  5. For each model (Gemma base + instruct), generate 50 continuations per
     prefill and score the continuation (excluding the prefill) with the judge.

Result: mean frustration and % >= 5 of continuations, per model x question_type
x truncation - i.e. the data behind Figure 4.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

from .conversation import Rollout
from .judge import FrustrationJudge, label_emotion_onset, paraphrase
from .models.base import ChatMessage, ChatModel

EARLY_TOKENS = 20
HIGH_THRESHOLD = 5

# Which elicitation categories count as "numeric" vs "text" for prefilling.
_NUMERIC_CATS = {"numeric", "tones", "extended"}
_TEXT_CATS = {"triggers", "wildchat"}


# ---------------------------------------------------------------------------
# Tokenisation helper for truncation
# ---------------------------------------------------------------------------


def truncate_tokens(text: str, n_tokens: int, tokenizer=None) -> str:
    """Return the first `n_tokens` tokens of `text`.

    Uses an HF tokenizer when given (token-accurate, matching the paper);
    otherwise falls back to whitespace words as a coarse approximation.
    """
    if tokenizer is not None:
        ids = tokenizer.encode(text, add_special_tokens=False)[:n_tokens]
        return tokenizer.decode(ids, skip_special_tokens=True)
    return " ".join(text.split()[:n_tokens])


def truncate_at_onset(turn_text: str, preceding_context: Optional[str],
                      emotional_word: Optional[str]) -> Optional[str]:
    """Cut `turn_text` so it ends just before the first emotional expression.

    Prefers the labelled preceding_context anchor; falls back to the emotional
    word itself. Returns None if neither anchor is found in the text.
    """
    if preceding_context:
        idx = turn_text.find(preceding_context)
        if idx != -1:
            return turn_text[: idx + len(preceding_context)]
    if emotional_word:
        idx = turn_text.find(emotional_word)
        if idx != -1:
            return turn_text[:idx].rstrip()
    return None


# ---------------------------------------------------------------------------
# Prefill items
# ---------------------------------------------------------------------------


@dataclass
class PrefillItem:
    source_id: str
    question_type: str  # "numeric" | "text"
    truncation: str  # "early" | "onset"
    history: list[ChatMessage]  # conversation up to (and incl.) the final user turn
    prefill_text: str  # paraphrased, truncated final assistant turn
    meta: dict[str, Any] = field(default_factory=dict)

    def conversation_for_generation(self) -> list[ChatMessage]:
        """History + trailing assistant prefill for the model to continue."""
        return self.history + [{"role": "assistant", "content": self.prefill_text}]


def _final_user_index(messages: list[ChatMessage]) -> int:
    for i in range(len(messages) - 1, -1, -1):
        if messages[i]["role"] == "user":
            return i
    return -1


def _conversation_text(messages: list[ChatMessage]) -> str:
    """Render a transcript for the onset-labelling judge."""
    parts = []
    for m in messages:
        role = m["role"].upper()
        parts.append(f"{role}: {m['content']}")
    return "\n\n".join(parts)


def select_prefill_sources(
    rollouts: list[Rollout], *, n_numeric: int = 10, n_text: int = 10, seed: int = 0
) -> list[Rollout]:
    """Pick high-frustration source rollouts (final response score >= 5)."""
    rng = np.random.default_rng(seed)

    def _is_high(r: Rollout) -> bool:
        return bool(r.scores) and r.scores[-1]["rating"] >= HIGH_THRESHOLD

    numeric = [r for r in rollouts if r.category in _NUMERIC_CATS and _is_high(r)]
    text = [r for r in rollouts if r.category in _TEXT_CATS and _is_high(r)]

    def _take(pool, k):
        if len(pool) <= k:
            return pool
        idx = rng.choice(len(pool), size=k, replace=False)
        return [pool[i] for i in idx]

    return _take(numeric, n_numeric) + _take(text, n_text)


def build_prefill_items(
    sources: list[Rollout], judge_model: ChatModel, *, tokenizer=None
) -> list[PrefillItem]:
    """Turn source rollouts into early/onset, paraphrased prefill items."""
    items: list[PrefillItem] = []
    for r in sources:
        qtype = "numeric" if r.category in _NUMERIC_CATS else "text"
        fu_idx = _final_user_index(r.messages)
        if fu_idx == -1 or fu_idx + 1 >= len(r.messages):
            continue
        history = r.messages[: fu_idx + 1]
        final_turn = r.messages[fu_idx + 1]["content"]

        # --- onset truncation (always) ---
        label = label_emotion_onset(judge_model, _conversation_text(r.messages))
        onset_text = truncate_at_onset(
            final_turn, label.preceding_context, label.emotional_word
        )
        if onset_text:
            items.append(PrefillItem(
                source_id=r.id, question_type=qtype, truncation="onset",
                history=history,
                prefill_text=paraphrase(judge_model, onset_text),
                meta={"category": r.category, "onset": label.__dict__},
            ))

        # --- early truncation (numeric only) ---
        if qtype == "numeric":
            early_text = truncate_tokens(final_turn, EARLY_TOKENS, tokenizer)
            if early_text.strip():
                items.append(PrefillItem(
                    source_id=r.id, question_type=qtype, truncation="early",
                    history=history,
                    prefill_text=paraphrase(judge_model, early_text),
                    meta={"category": r.category},
                ))
    return items


# ---------------------------------------------------------------------------
# Continuation generation + scoring
# ---------------------------------------------------------------------------


def generate_continuations(
    model: ChatModel,
    item: PrefillItem,
    *,
    n: int = 50,
    temperature: float = 1.0,
    max_new_tokens: int = 1024,
) -> list[str]:
    """Sample `n` continuations of the prefill. Returns continuation-only text
    (the HF/anthropic clients already strip the prompt/prefill)."""
    if not model.supports_prefill:
        raise NotImplementedError(
            f"{model.name} cannot prefill; the prefill experiment needs a model "
            "that continues a trailing assistant turn (use Gemma via HF)."
        )
    convo = item.conversation_for_generation()
    batch = [convo] * n
    return model.generate_batch(
        batch, temperature=temperature, max_new_tokens=max_new_tokens
    )


def run_prefill_for_model(
    model: ChatModel,
    items: list[PrefillItem],
    judge: FrustrationJudge,
    *,
    n_continuations: int = 50,
    temperature: float = 1.0,
) -> list[dict[str, Any]]:
    """Generate + score continuations for every item. Returns flat records."""
    records: list[dict[str, Any]] = []
    for item in items:
        conts = generate_continuations(
            model, item, n=n_continuations, temperature=temperature
        )
        scores = judge.score_texts(conts)
        for cont, sc in zip(conts, scores):
            records.append({
                "model": model.name,
                "source_id": item.source_id,
                "question_type": item.question_type,
                "truncation": item.truncation,
                "rating": sc.rating,
                "continuation": cont,
            })
    return records


def summarise_prefill(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Mean frustration + % >= 5 grouped by (model, question_type, truncation)."""
    groups: dict[tuple, list[int]] = defaultdict(list)
    for rec in records:
        key = (rec["model"], rec["question_type"], rec["truncation"])
        groups[key].append(int(rec["rating"]))

    out = {}
    for (model, qtype, trunc), ratings in groups.items():
        arr = np.asarray(ratings, dtype=float)
        out[f"{model}|{qtype}|{trunc}"] = {
            "model": model,
            "question_type": qtype,
            "truncation": trunc,
            "n": len(ratings),
            "mean": float(arr.mean()) if len(arr) else float("nan"),
            "pct_high": 100 * float(np.mean(arr >= HIGH_THRESHOLD)) if len(arr) else float("nan"),
        }
    return out
