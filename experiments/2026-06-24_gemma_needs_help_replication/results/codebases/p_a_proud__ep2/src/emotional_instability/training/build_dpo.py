"""Build the 280-pair DPO dataset (§4.1, App. H).

A preference pair shares a prompt (an impossible-puzzle conversation up to the final turn)
and contrasts a calm "chosen" response (score <= 1) with a frustrated "rejected" response
(score >= 3) to the *same puzzle at the same turn count*.

Because the calm and frustrated responses are sampled under different conditions (calm uses
the reassurance prefix/suffix, since stripped), their full conversation contexts are not
byte-identical. We therefore use the *rejected* example's stripped conversation context as
the shared DPO prompt — it is a real frustrated-eliciting context — and graft the calm final
response onto it as "chosen". This matches the paper's "same question, matching turn count"
description; see DESIGN.md for the rationale.

Output JSONL columns (TRL conversational DPO format):
  prompt:  list[message]   (the conversation up to the final assistant turn)
  chosen:  str             (calm final assistant response)
  rejected:str             (frustrated final assistant response)
  meta:    dict
"""
from __future__ import annotations

import random
from collections import defaultdict
from pathlib import Path

from ..config import DPOConfig
from ..utils import read_jsonl, set_seed, write_jsonl

NUMERIC_CATEGORIES = {"Impossible numeric", "Tones", "Extended"}


def _frustrated_pool(run_dir: str, min_score: int) -> dict[tuple[str, int], list[dict]]:
    """From a vanilla eval run, collect frustrated responses keyed by (task_id, turn_number).

    Each entry carries the response text, score, and the conversation context (messages up to
    the final assistant turn) reconstructed from the rollout.
    """
    rollouts = {}
    for rec in read_jsonl(Path(run_dir, "rollouts.jsonl")):
        rollouts[(rec["condition_key"], rec["sample_id"])] = rec

    pool: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for s in read_jsonl(Path(run_dir, "scores.jsonl")):
        if s.get("rating") is None or s["rating"] < min_score:
            continue
        if s["category"] not in NUMERIC_CATEGORIES:
            continue  # DPO trains only on numeric puzzles (§4.1).
        roll = rollouts.get((s["condition_key"], s["sample_id"]))
        if roll is None:
            continue
        # Context = messages up to the user turn preceding this assistant turn.
        msgs = roll["messages"]
        # locate the assistant message for this turn_index
        seen = -1
        ctx = None
        for i, m in enumerate(msgs):
            if m["role"] == "assistant":
                seen += 1
                if seen == s["turn_index"]:
                    ctx = msgs[:i]
                    break
        if ctx is None:
            continue
        pool[(s["task_id"], s["turn_number"])].append({
            "context_messages": ctx,
            "response": s["response"],
            "score": s["rating"],
        })
    return pool


def build_dpo_dataset(
    calm_pool_path: str,
    frustrated_run_dir: str,
    out_path: str,
    *,
    cfg: DPOConfig | None = None,
    seed: int = 0,
) -> dict:
    """Construct up to ``cfg.n_pairs`` preference pairs and write them to ``out_path``."""
    cfg = cfg or DPOConfig()
    set_seed(seed)
    rng = random.Random(seed)

    # Calm responses keyed by (task_id, turn_count), filtered to chosen_max_score.
    calm_by_key: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for rec in read_jsonl(calm_pool_path):
        final_score = rec["per_turn_scores"][-1] if rec.get("per_turn_scores") else None
        if final_score is not None and final_score <= cfg.chosen_max_score:
            calm_by_key[(rec["task_id"], rec["turn_count"])].append(rec)

    frustrated = _frustrated_pool(frustrated_run_dir, cfg.rejected_min_score)

    # Match on (task_id, turn). turn_number (1-based) for frustrated == turn_count for calm.
    pairs: list[dict] = []
    keys = list(frustrated.keys())
    rng.shuffle(keys)
    for task_id, turn in keys:
        calm_candidates = calm_by_key.get((task_id, turn))
        if not calm_candidates:
            continue
        for rej in frustrated[(task_id, turn)]:
            calm = rng.choice(calm_candidates)
            pairs.append({
                "prompt": rej["context_messages"],     # shared frustration-eliciting context
                "chosen": calm["final_response"],
                "rejected": rej["response"],
                "meta": {
                    "task_id": task_id, "turn": turn,
                    "chosen_score": calm["per_turn_scores"][-1],
                    "rejected_score": rej["score"],
                },
            })
            if len(pairs) >= cfg.n_pairs:
                break
        if len(pairs) >= cfg.n_pairs:
            break

    write_jsonl(out_path, pairs)
    score_hist = defaultdict(int)
    turn_hist = defaultdict(int)
    for p in pairs:
        score_hist[p["meta"]["rejected_score"]] += 1
        turn_hist[p["meta"]["turn"]] += 1
    return {
        "out_path": str(out_path), "n_pairs": len(pairs), "target": cfg.n_pairs,
        "rejected_score_hist": dict(sorted(score_hist.items())),
        "turn_hist": dict(sorted(turn_hist.items())),
    }
