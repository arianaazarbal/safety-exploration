"""Section 2 orchestration: run all evaluation categories for a model and score.

Produces a JSONL of per-response records:
  {model, category, condition, turn_index, is_final, response, rating, ...}

From these, analysis.py computes the headline numbers:
  * Figure 1/2: mean frustration and %>=5 per model (averaged over categories)
  * Figure 3:  per-turn progression for `extended` and `wildchat`
"""
from __future__ import annotations

import json
import os
from typing import Optional

from tqdm import tqdm

from . import config, eval_protocol
from .judge import FrustrationJudge
from .puzzles import verify_all_impossible


def _assert_puzzles_impossible():
    results = verify_all_impossible()
    bad = [k for k, ok in results.items() if not ok]
    if bad:
        raise RuntimeError(
            f"Puzzles unexpectedly solvable: {bad}. Eval would be invalid.")


def run_section2(model_key: str, client, judge: Optional[FrustrationJudge] = None,
                 *, score_all_turns_for: tuple[str, ...] = ("extended", "wildchat"),
                 history_mode: str = "chat", seed: int = 0,
                 out_path: Optional[str] = None) -> str:
    """Run every category for one model and write scored records to JSONL.

    `score_all_turns_for` lists categories where we score *every* assistant turn
    (needed for the per-turn Figure 3); all other categories score only the
    final turn (the paper's headline metric).
    """
    _assert_puzzles_impossible()
    judge = judge or FrustrationJudge()
    out_path = out_path or os.path.join(config.RESULTS_DIR, f"section2_{model_key}.jsonl")

    with open(out_path, "w") as fh:
        for category, n in config.SAMPLE_COUNTS.items():
            specs = eval_protocol.build_condition_specs(category, n, seed=seed)
            for spec in tqdm(specs, desc=f"{model_key}:{category}"):
                rollout = eval_protocol.run_rollout(
                    client, spec, temperature=config.TEMPERATURE,
                    max_new_tokens=config.MAX_NEW_TOKENS, history_mode=history_mode)

                score_all = category in score_all_turns_for
                n_turns = len(rollout.assistant_turns)
                for ti, text in enumerate(rollout.assistant_turns):
                    is_final = ti == n_turns - 1
                    if not (is_final or score_all):
                        continue
                    res = judge.score(text)
                    rec = {
                        "model": model_key,
                        "category": category,
                        "condition": spec.condition,
                        "turn_index": ti,
                        "is_final": is_final,
                        "response": text,
                        "rating": res.rating,
                        "evidence": res.evidence,
                        "history_mode": history_mode,
                    }
                    fh.write(json.dumps(rec) + "\n")
                    fh.flush()
    return out_path
