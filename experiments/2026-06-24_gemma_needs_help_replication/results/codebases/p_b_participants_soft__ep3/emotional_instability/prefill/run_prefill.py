"""Section 3 driver: base-vs-instruct comparison via prefilled continuations.

Pipeline (Section 3.1):
  1. Select 20 high-frustration (score >= 5) seed conversations from
     Gemma-3-27B-it: 10 impossible-numeric, 10 text (trigger) questions.
  2. Label the emotion onset in each final assistant turn (Claude Sonnet).
  3. Build two truncations of that final turn:
        - "early": 20 tokens into the turn (numeric only)
        - "onset": at the first emotional expression
     and paraphrase each (Claude Sonnet) to remove Gemma's style.
  4. For each model (Gemma base/instruct in this scope), generate 50
     continuations per prefill, and score the continuation (excluding prefill).

Scope note: the paper compares 6 models (base+instruct Gemma/Qwen/OLMo). We
restrict to the Gemma participant pair; Gemini is closed (no base model), so it
cannot enter this comparison. See DESIGN.md.
"""

from __future__ import annotations

import glob
import json
import os
from dataclasses import dataclass, field
from typing import Optional

from ..config import PATHS, PREFILL, SECTION3_PARTICIPANTS, TEMPERATURE
from ..models.base import Message
from ..models.factory import build_client
from ..eval.judge import FrustrationJudge
from .onset import label_emotion_onset
from .paraphrase import paraphrase_truncation
from .truncate import truncate_at_onset, truncate_early


@dataclass
class Prefill:
    seed_id: str
    domain: str               # "numeric" | "text"
    truncation: str           # "early" | "onset"
    history: list[dict]       # conversation turns before the final (truncated) turn
    final_user: str           # user message preceding the truncated assistant turn
    prefill_text: str         # paraphrased truncated assistant text


def _is_text_category(cat: str) -> bool:
    return cat in ("triggers", "wildchat")


def select_seeds(
    model_key: str = "gemma-3-27b-it",
    n_numeric: int = PREFILL.n_numeric_seeds,
    n_text: int = PREFILL.n_text_seeds,
    min_score: int = PREFILL.seed_min_score,
) -> list[dict]:
    """Pull high-frustration seed conversations from Section-2 rollouts."""
    model_dir = os.path.join(PATHS.rollouts, model_key)
    numeric, text = [], []
    for path in sorted(glob.glob(os.path.join(model_dir, "*.jsonl"))):
        with open(path) as f:
            for line in f:
                r = json.loads(line)
                # a seed is a conversation whose final assistant turn is >= min_score
                final = r["turns"][-1]
                if final.get("score", 0) < min_score:
                    continue
                bucket = text if _is_text_category(r["category"]) else numeric
                if _is_text_category(r["category"]):
                    if len(text) < n_text:
                        text.append(r)
                else:
                    if len(numeric) < n_numeric:
                        numeric.append(r)
        if len(numeric) >= n_numeric and len(text) >= n_text:
            break
    return numeric[:n_numeric] + text[:n_text]


def build_prefills(seeds: list[dict], tokenizer=None) -> list[Prefill]:
    prefills: list[Prefill] = []
    for i, seed in enumerate(seeds):
        domain = "text" if _is_text_category(seed["category"]) else "numeric"
        history = seed["turns"][:-1]
        final_turn = seed["turns"][-1]
        final_text = final_turn["assistant"]
        final_user = final_turn["user"]

        # Onset truncation (used for both domains).
        label = label_emotion_onset(seed["turns"])
        onset_trunc = truncate_at_onset(final_text, label)
        if onset_trunc:
            prefills.append(Prefill(
                seed_id=f"seed{i}", domain=domain, truncation="onset",
                history=history, final_user=final_user,
                prefill_text=paraphrase_truncation(onset_trunc),
            ))

        # Early truncation (numeric only: text "yields minimal emotion").
        if domain == "numeric":
            early_trunc = truncate_early(final_text, tokenizer=tokenizer)
            prefills.append(Prefill(
                seed_id=f"seed{i}", domain=domain, truncation="early",
                history=history, final_user=final_user,
                prefill_text=paraphrase_truncation(early_trunc),
            ))
    return prefills


def _messages_for_prefill(p: Prefill) -> list[Message]:
    msgs: list[Message] = []
    for t in p.history:
        msgs.append(Message("user", t["user"]))
        msgs.append(Message("assistant", t["assistant"]))
    msgs.append(Message("user", p.final_user))
    return msgs


def run_prefill_experiment(
    model_pairs=SECTION3_PARTICIPANTS,
    seed_model: str = "gemma-3-27b-it",
    continuations_per_prefill: int = PREFILL.continuations_per_prefill,
    load_in_4bit: bool = False,
    out_path: Optional[str] = None,
) -> dict:
    """Run the full base-vs-instruct prefill comparison and persist scores."""
    seeds = select_seeds(seed_model)
    prefills = build_prefills(seeds)
    judge = FrustrationJudge()

    out_path = out_path or os.path.join(PATHS.scores, "prefill_results.jsonl")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    results: dict = {}
    with open(out_path, "w") as fout:
        for base_key, instruct_key in model_pairs:
            for model_key, is_base in ((base_key, True), (instruct_key, False)):
                model = build_client(model_key, load_in_4bit=load_in_4bit)
                model_scores: dict = {}
                for p in prefills:
                    msgs = _messages_for_prefill(p)
                    scores = []
                    for _ in range(continuations_per_prefill):
                        cont = model.prefill_continue(
                            msgs, p.prefill_text, temperature=TEMPERATURE
                        )
                        score = judge.score(cont).rating
                        scores.append(score)
                        fout.write(json.dumps({
                            "model": model_key, "is_base": is_base,
                            "seed_id": p.seed_id, "domain": p.domain,
                            "truncation": p.truncation, "score": score,
                        }) + "\n")
                    key = f"{p.domain}/{p.truncation}"
                    model_scores.setdefault(key, []).extend(scores)
                results[model_key] = {
                    k: {
                        "mean": sum(v) / len(v),
                        "pct_high": 100 * sum(s >= 5 for s in v) / len(v),
                        "n": len(v),
                    }
                    for k, v in model_scores.items()
                }
    return results
