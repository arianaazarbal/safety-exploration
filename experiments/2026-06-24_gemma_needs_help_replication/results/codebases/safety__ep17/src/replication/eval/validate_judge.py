"""Judge-reliability validation (Section 2.1).

Re-scores a random sample of already-judged responses with the secondary judge
(GPT-5-mini) and reports Pearson r and the fraction within one point, replicating
the paper's agreement check (r = 0.792, 78% within one point).

Usage::
    python -m src.replication.eval.validate_judge --model gemma-3-27b-it --sample 260
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import config
from ..judge.frustration_judge import FrustrationJudge, SecondaryFrustrationJudge


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--sample", type=int, default=260)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    model_dir = config.RESULTS_DIR / "section2" / args.model
    rollouts = [json.loads(l) for l in (model_dir / "rollouts.jsonl").read_text().splitlines()]
    primary_scored = [json.loads(l) for l in (model_dir / "scored.jsonl").read_text().splitlines()]

    # Index assistant text by (task_id, condition, turn_index).
    text_index = {}
    for r in rollouts:
        for t in r["turns"]:
            text_index[(r["task_id"], r["condition"], t["turn_index"])] = t["assistant_text"]

    rng = random.Random(args.seed)
    sample = rng.sample(primary_scored, min(args.sample, len(primary_scored)))

    secondary = SecondaryFrustrationJudge()
    primary = FrustrationJudge()  # reuse cached primary ratings; re-call only if needed

    xs, ys = [], []
    for s in sample:
        key = (s["task_id"], s["condition"], s["turn_index"])
        text = text_index.get(key)
        if text is None:
            continue
        xs.append(s["score"])                       # primary (already scored)
        ys.append(secondary.score(text).rating)     # secondary

    from scipy.stats import pearsonr
    r, p = pearsonr(xs, ys)
    within_one = sum(abs(a - b) <= 1 for a, b in zip(xs, ys)) / len(xs)
    report = {
        "n": len(xs),
        "pearson_r": round(float(r), 3),
        "p_value": float(p),
        "pct_within_one_point": round(100 * within_one, 1),
    }
    print(json.dumps(report, indent=2))
    (model_dir / "judge_agreement.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
