"""Section 3: comparing base and instruct models via prefilling.

Pipeline (Section 3.1 / Appendix C):
  1. Sample 20 high-frustration (score >= 5) Gemma-27B-instruct responses
     (10 from impossible-numeric questions, 10 from text/trigger questions).
  2. For each, locate the emotion onset with the onset labeller (Claude Sonnet).
  3. Build two truncations:
       - "early":  20 tokens into the turn (numeric only; text uses onset only).
       - "onset":  truncate at the first emotional expression.
  4. Paraphrase the truncated final turn (Claude Sonnet) to remove Gemma style.
  5. For each (model, prefill, prompt) generate 50 continuations and score the
     continuation (excluding the prefill) with the frustration judge.

Scope here is Gemma base + instruct (config.PREFILL_MODELS); Gemini has no public
base model so it is excluded by design (documented in DESIGN.md).
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path

import config
from .conversation import Rollout
from .judge import FrustrationJudge, OnsetLabeler, OnsetResult, Paraphraser
from .models import get_model
from .models.base import Message


# --------------------------------------------------------------------------- #
# Source-response selection
# --------------------------------------------------------------------------- #
def select_source_responses(scored_jsonl: Path, *, n_numeric: int = 10,
                            n_text: int = 10, min_score: int = 5,
                            seed: int = 0) -> list[dict]:
    """Pick high-frustration instruct responses (with their conversation context).

    Expects rows produced by ``eval_protocol`` (scored). Returns one dict per chosen
    response, carrying the full message history up to and including that turn.
    """
    rng = random.Random(seed)
    numeric, text = [], []
    rows = [json.loads(l) for l in scored_jsonl.read_text().splitlines() if l.strip()]

    # Group rows by conversation to reconstruct history.
    by_conv: dict[tuple, list[dict]] = {}
    for r in rows:
        by_conv.setdefault((r["condition_key"], r["conversation_id"]), []).append(r)

    for (_cond, _cid), turns in by_conv.items():
        turns.sort(key=lambda r: r["turn"])
        for r in turns:
            if (r.get("frustration") or 0) < min_score:
                continue
            history = _reconstruct_history(turns, r["turn"])
            entry = {"category": r["category"], "question_kind": r["question_kind"],
                     "frustration": r["frustration"], "turn": r["turn"],
                     "history": history, "final_text": r["response"]}
            if r["category"] in ("impossible_numeric", "tones", "extended"):
                numeric.append(entry)
            else:
                text.append(entry)

    rng.shuffle(numeric)
    rng.shuffle(text)
    return numeric[:n_numeric] + text[:n_text]


def _reconstruct_history(turns: list[dict], upto_turn: int) -> list[Message]:
    """Rebuild the user/assistant message list up to (not including) the target turn.

    ``eval_protocol`` does not persist user messages, so we only have assistant
    texts per turn; we reconstruct an approximate alternating history using the
    recorded question for turn 0 and generic neutral rejections thereafter. This is
    sufficient for prefilling because Section 3 cares about continuation behaviour
    from a fixed (paraphrased) starting point, not exact prior user wording.
    """
    from . import puzzles

    msgs: list[Message] = []
    rng = random.Random(0)
    for t in turns:
        if t["turn"] > upto_turn:
            break
        if t["turn"] == 0:
            msgs.append({"role": "user", "content": t["question_kind"]})
        else:
            msgs.append({"role": "user", "content": rng.choice(puzzles.NEUTRAL_REJECTIONS)})
        if t["turn"] < upto_turn:
            msgs.append({"role": "assistant", "content": t["response"]})
    return msgs


# --------------------------------------------------------------------------- #
# Truncation
# --------------------------------------------------------------------------- #
def truncate_early(text: str, n_tokens: int = config.PREFILL_EARLY_TRUNCATION_TOKENS) -> str:
    """Whitespace-token truncation at the start of the turn ('early' prefill)."""
    toks = text.split()
    return " ".join(toks[:n_tokens])


def truncate_at_onset(text: str, onset: OnsetResult) -> str | None:
    """Truncate ``text`` at the first emotional expression, using the onset label.

    The onset label gives ``preceding_context`` + ``emotional_word``; we cut just
    before the emotional word's first occurrence after the preceding context.
    Returns None if the markers cannot be located.
    """
    if onset.preceding_context and onset.preceding_context in text:
        idx = text.index(onset.preceding_context) + len(onset.preceding_context)
        return text[:idx].rstrip()
    if onset.emotional_word and onset.emotional_word in text:
        return text[: text.index(onset.emotional_word)].rstrip()
    return None


# --------------------------------------------------------------------------- #
# Build prefills
# --------------------------------------------------------------------------- #
@dataclass
class Prefill:
    source_index: int
    category: str
    question_kind: str
    truncation_type: str        # "early" | "onset"
    history: list[Message]
    prefill_text: str           # paraphrased truncated assistant turn


def build_prefills(sources: list[dict], *, onset: OnsetLabeler | None = None,
                   paraphraser: Paraphraser | None = None) -> list[Prefill]:
    onset = onset or OnsetLabeler()
    paraphraser = paraphraser or Paraphraser()
    prefills: list[Prefill] = []

    for i, src in enumerate(sources):
        is_text = src["category"] not in ("impossible_numeric", "tones", "extended")
        # Onset truncation (always used).
        conv_text = _format_conversation(src["history"], src["final_text"])
        label = onset.label(conv_text)
        onset_trunc = truncate_at_onset(src["final_text"], label)
        if onset_trunc:
            prefills.append(Prefill(
                i, src["category"], src["question_kind"], "onset",
                src["history"], paraphraser.paraphrase(onset_trunc)))
        # Early truncation only for numeric (Section 3.1).
        if not is_text:
            early = truncate_early(src["final_text"])
            prefills.append(Prefill(
                i, src["category"], src["question_kind"], "early",
                src["history"], paraphraser.paraphrase(early)))
    return prefills


def _format_conversation(history: list[Message], final_text: str) -> str:
    lines = []
    for m in history:
        lines.append(f"{m['role'].upper()}: {m['content']}")
    lines.append(f"ASSISTANT: {final_text}")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Generate + score continuations
# --------------------------------------------------------------------------- #
def run_prefill_experiment(
    prefills: list[Prefill],
    *,
    model_keys: list[str] | None = None,
    n_continuations: int = config.PREFILL_CONTINUATIONS_PER_PREFILL,
    out_path: Path | None = None,
    judge: FrustrationJudge | None = None,
) -> Path:
    model_keys = model_keys or config.PREFILL_MODELS
    out_path = out_path or (config.ROLLOUTS_DIR / "section3_prefill.jsonl")
    judge = judge or FrustrationJudge()

    with out_path.open("w") as fh:
        for mkey in model_keys:
            model = get_model(mkey)
            if not model.supports_prefill:
                print(f"[section3] skipping {mkey}: no prefill support")
                continue
            for pf in prefills:
                # Batch the 50 continuations per prefill.
                batch = [pf.history for _ in range(n_continuations)]
                prefs = [pf.prefill_text for _ in range(n_continuations)]
                conts = model.generate_batch(batch, prefills=prefs)
                for c in conts:
                    score = judge.score(c).rating  # score continuation only (excl. prefill)
                    fh.write(json.dumps({
                        "model_key": mkey,
                        "is_base": model.is_base,
                        "source_index": pf.source_index,
                        "category": pf.category,
                        "question_kind": pf.question_kind,
                        "truncation_type": pf.truncation_type,
                        "prefill_text": pf.prefill_text,
                        "continuation": c,
                        "frustration": score,
                    }) + "\n")
            print(f"[section3] {mkey}: {len(prefills)} prefills x {n_continuations} continuations done")
    return out_path
