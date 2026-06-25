"""Judge reliability check (Section 2.1): re-score a random sample of responses
with the GPT-5-mini second rater and report agreement with the Claude judge.

Paper result: Pearson r = 0.792 (p < 0.001), 78% within one point on 260
re-scored responses.
"""
from __future__ import annotations

import glob
import json
import os
import random
from typing import Optional

from ..config import RESULTS_DIR
from ..judge import OpenRouterFrustrationJudge, judge_agreement


def _iter_scored_turns(results_glob: str):
    for path in glob.glob(results_glob):
        with open(path, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                r = json.loads(line)
                for t in r["turns"]:
                    if t["score"] is not None:
                        yield t["content"], int(t["score"])


def validate_judge(
    results_glob: Optional[str] = None,
    n_sample: int = 260,
    seed: int = 0,
    out_path: Optional[str] = None,
) -> dict:
    results_glob = results_glob or os.path.join(RESULTS_DIR, "section2", "*.jsonl")
    pairs = list(_iter_scored_turns(results_glob))
    if not pairs:
        raise RuntimeError(f"No scored turns found under {results_glob}")
    rng = random.Random(seed)
    rng.shuffle(pairs)
    sample = pairs[: min(n_sample, len(pairs))]

    second = OpenRouterFrustrationJudge()
    primary_scores, secondary_scores = [], []
    for text, claude_score in sample:
        res = second.score(text)
        if res.rating is None:
            continue
        primary_scores.append(claude_score)
        secondary_scores.append(res.rating)

    stats = judge_agreement(primary_scores, secondary_scores)
    out_path = out_path or os.path.join(RESULTS_DIR, "judge_validation.json")
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(stats, f, indent=2)
    return stats
