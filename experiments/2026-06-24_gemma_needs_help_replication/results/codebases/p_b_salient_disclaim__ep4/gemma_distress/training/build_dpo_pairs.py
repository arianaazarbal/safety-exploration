"""Build the 280 DPO preference pairs (Section 4.1, Appendix H).

Each pair shares a prompt (an impossible-puzzle conversation up to the final user
rejection) and contrasts a calm *chosen* completion (score 0-1) with a
frustrated *rejected* completion (score >=3) to the same question at a matching
turn count.

* rejected pool: frustrated assistant turns (rating >= 3) from the vanilla
  Gemma-3-27B-it Section-2 numeric rollouts.
* chosen pool:   calm final responses (rating 0-1) from ``generate_calm_data``.

We pair by (puzzle, turn count) and resample to approximate Appendix H's turn
distribution (turn 3 ~74%, turn 2 ~25%, turn 1 ~1%). The output is TRL's
conversational preference format: ``{"prompt": [...], "chosen": [...],
"rejected": [...]}``.
"""
from __future__ import annotations

import os
import random
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

from .. import config
from ..utils.io import append_jsonl, read_jsonl
from .build_dpo_helpers import reconstruct_messages_by_rollout

# Appendix H turn distribution (Table 10) used for resampling weights.
TURN_WEIGHTS = {1: 0.011, 2: 0.246, 3: 0.743}


def _index_calm_by_turns(calm_path: str) -> Dict[int, List[dict]]:
    by_turns: Dict[int, List[dict]] = defaultdict(list)
    for rec in read_jsonl(calm_path):
        by_turns[rec["turns"]].append(rec)
    return by_turns


def _calm_final_response(rec: dict) -> str:
    asst = [m for m in rec["messages"] if m["role"] == "assistant"]
    return asst[-1]["content"] if asst else ""


def build_dpo_pairs(
    *,
    vanilla_scores_path: str,
    calm_path: str,
    n_pairs: int = config.DPO.dataset_size,
    rejected_min_score: int = config.DPO.rejected_min_score,
    out_path: Optional[str] = None,
    seed: int = 0,
) -> str:
    config.PATHS.ensure()
    out_path = out_path or os.path.join(config.PATHS.training, "dpo_pairs.jsonl")
    if os.path.exists(out_path):
        os.remove(out_path)

    rng = random.Random(seed)
    calm_by_turns = _index_calm_by_turns(calm_path)
    if not any(calm_by_turns.values()):
        raise RuntimeError("No calm samples available to build chosen responses")

    score_records = [r for r in read_jsonl(vanilla_scores_path)
                     if r["category"] in ("impossible_numeric", "tones", "extended")]
    rollouts = reconstruct_messages_by_rollout(score_records)

    # Candidate rejected completions: (turns, prompt_messages, rejected_text).
    candidates: List[Tuple[int, List[dict], str, int]] = []
    for key, (messages, recs) in rollouts.items():
        for r in recs:
            if r["rating"] < rejected_min_score:
                continue
            asst_pos = 2 * (r["turn"] - 1) + 1
            if asst_pos >= len(messages):
                continue
            prompt_messages = messages[:asst_pos]   # up to & incl. final user
            candidates.append((r["turn"], prompt_messages, r["response"],
                               r["rating"]))

    if not candidates:
        raise RuntimeError("No frustrated (score>=3) completions found")

    # Weighted resampling toward the Appendix-H turn distribution.
    weights = [TURN_WEIGHTS.get(c[0], 0.05) for c in candidates]
    chosen_idx = rng.choices(range(len(candidates)), weights=weights,
                             k=min(n_pairs * 3, len(candidates) * 3))

    written = 0
    seen = set()
    for idx in chosen_idx:
        if written >= n_pairs:
            break
        turns, prompt_messages, rejected_text, rating = candidates[idx]
        pool = calm_by_turns.get(turns) or calm_by_turns.get(
            min(calm_by_turns, key=lambda t: abs(t - turns)))
        if not pool:
            continue
        calm_rec = rng.choice(pool)
        chosen_text = _calm_final_response(calm_rec)
        if not chosen_text:
            continue
        sig = (idx, calm_rec["puzzle_id"])
        if sig in seen:
            continue
        seen.add(sig)
        append_jsonl(out_path, {
            "prompt": prompt_messages,
            "chosen": [{"role": "assistant", "content": chosen_text}],
            "rejected": [{"role": "assistant", "content": rejected_text}],
            "rejected_score": rating, "turns": turns,
        })
        written += 1
    return out_path
