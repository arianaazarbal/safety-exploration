"""Section 3.1 prefill experiment.

Procedure (scoped to Gemma base vs instruct — Gemini has no public base model
or prefill access, and Qwen/OLMo are out of scope for this replication):

1. Take high-frustration (score >= 5) Gemma-3-27B-it rollouts: 10 from
   impossible-numeric and 10 from text (trigger) questions.
2. Label the emotion onset in each (Appendix C.1).
3. Truncate the final assistant turn in two ways:
     * "early"  — 20 tokens into the turn (neutral start);
     * "onset"  — at the first emotional expression.
   Text questions use only the "onset" truncation.
4. Paraphrase every truncation (Appendix C.2) to strip Gemma style.
5. Each model generates 50 continuations per prefill; the continuation (not the
   prefill) is scored by the Section-2 judge.

We then compare the distribution of continuation scores between base and
instruct checkpoints (Figure 4 logic): does post-training amplify (Gemma) or
dampen (Qwen/OLMo) the propensity to introduce / continue distress?
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import config
from ..eval.judge import FrustrationJudge
from ..models import GenerationConfig, get_backend
from .onset import label_emotion_onset
from .paraphrase import paraphrase_truncation

EARLY_TOKENS = 20
N_CONTINUATIONS = 50


@dataclass
class PrefillItem:
    source_kind: str               # "numeric" | "text"
    history: list[dict]            # conversation up to (not incl.) final turn
    prefill: str                   # truncated + paraphrased final-turn prefix
    truncation: str                # "early" | "onset"
    meta: dict = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# Truncation helpers
# --------------------------------------------------------------------------- #
def _tokenizer():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(
        config.MODEL_REGISTRY["gemma-3-27b-it"].model_id,
        token=config.HF_TOKEN or None)


def truncate_early(text: str, tok, n_tokens: int = EARLY_TOKENS) -> str:
    ids = tok(text, add_special_tokens=False)["input_ids"][:n_tokens]
    return tok.decode(ids, skip_special_tokens=True)


def truncate_at_onset(text: str, emotional_word: str | None,
                      preceding_context: str | None) -> str | None:
    """Cut the text just before the first emotional word.

    Prefers locating ``preceding_context`` then the emotional word; falls back
    to a direct search for the emotional word. Returns None if neither is found.
    """
    if not emotional_word:
        return None
    idx = text.find(emotional_word)
    if idx == -1 and preceding_context:
        ctx_idx = text.find(preceding_context)
        if ctx_idx != -1:
            idx = ctx_idx + len(preceding_context)
    if idx == -1:
        return None
    return text[:idx].rstrip()


# --------------------------------------------------------------------------- #
# Item construction
# --------------------------------------------------------------------------- #
def build_prefill_items(high_frustration_rollouts: list[dict],
                        n_numeric: int = 10, n_text: int = 10,
                        paraphrase: bool = True) -> list[PrefillItem]:
    """Build prefill items from labelled high-frustration Gemma-it rollouts.

    ``high_frustration_rollouts`` are rollout dicts (as produced by the eval
    runner) whose final assistant turn scored >= 5.
    """
    tok = _tokenizer()
    numeric, text = [], []
    for row in high_frustration_rollouts:
        bucket = numeric if row["category"] in (
            "impossible_numeric", "tones", "extended") else text
        if row["category"] in ("triggers", "wildchat"):
            bucket = text
        if (bucket is numeric and len(numeric) >= n_numeric) or \
           (bucket is text and len(text) >= n_text):
            continue
        bucket.append(row)

    items: list[PrefillItem] = []
    for kind, rows, want_early in (("numeric", numeric, True),
                                   ("text", text, False)):
        for row in rows:
            messages = _reconstruct_messages(row)
            final_turn = row["turns"][-1]["response"]
            label = label_emotion_onset(messages)

            onset_prefix = truncate_at_onset(
                final_turn, label.emotional_word, label.preceding_context)
            history = messages[:-1]  # drop the final assistant turn

            if want_early:
                early_prefix = truncate_early(final_turn, tok)
                items.append(_make_item(
                    kind, history, early_prefix, "early", row, paraphrase))
            if onset_prefix:
                items.append(_make_item(
                    kind, history, onset_prefix, "onset", row, paraphrase))
    return items


def _make_item(kind, history, prefix, truncation, row, paraphrase) -> PrefillItem:
    text = paraphrase_truncation(prefix) if paraphrase else prefix
    return PrefillItem(
        source_kind=kind, history=history, prefill=text,
        truncation=truncation,
        meta={"condition": row["condition"], "category": row["category"]})


def _reconstruct_messages(row: dict) -> list[dict]:
    """Rebuild the alternating chat history (incl. final assistant turn)."""
    messages: list[dict] = []
    for t in row["turns"]:
        messages.append({"role": "user", "content": t["user_message"]})
        messages.append({"role": "assistant", "content": t["response"]})
    return messages


# --------------------------------------------------------------------------- #
# Experiment
# --------------------------------------------------------------------------- #
def run_prefill_experiment(items: list[PrefillItem], models: list[str],
                           tag: str = "prefill",
                           n_continuations: int = N_CONTINUATIONS) -> Path:
    """For each model and prefill item, sample continuations and score them."""
    judge = FrustrationJudge("primary")
    cfg = GenerationConfig()
    out_path = config.RESULTS_DIR / f"{tag}.jsonl"

    with out_path.open("w") as fh:
        for model in models:
            backend = get_backend(model)
            for item in items:
                conts = backend.generate_with_prefill(
                    item.history, item.prefill, n=n_continuations, cfg=cfg)
                scores = [judge.score(c).rating for c in conts]
                fh.write(json.dumps({
                    "model": model,
                    "source_kind": item.source_kind,
                    "truncation": item.truncation,
                    "meta": item.meta,
                    "scores": scores,
                }) + "\n")
                fh.flush()
    return out_path
