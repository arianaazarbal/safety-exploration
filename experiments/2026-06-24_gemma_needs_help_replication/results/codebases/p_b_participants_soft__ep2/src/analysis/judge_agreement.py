"""Judge-reliability validation (Section 2.1).

Randomly sample 260 already-scored responses, re-score them with GPT-5-mini using
the same prompt, and report Pearson r and the fraction within one point. The
paper reports r = 0.792 (p < 0.001) and 78% within one point.
"""
from __future__ import annotations

import argparse
import json
import random

import numpy as np
from scipy.stats import pearsonr

from ..config import CFG
from ..eval.judge import score_response


def collect_scored(models: list[str]) -> list[tuple[str, int]]:
    items = []
    for m in models:
        path = CFG.out("section2", f"{m}.jsonl")
        try:
            with open(path) as f:
                for line in f:
                    r = json.loads(line)
                    for t in r["turns"]:
                        if "score" in t and t["response"].strip():
                            items.append((t["response"], t["score"]))
        except FileNotFoundError:
            continue
    return items


def run(models: list[str], n: int = 260, seed: int = 0) -> dict:
    items = collect_scored(models)
    rng = random.Random(seed)
    sample = rng.sample(items, min(n, len(items)))

    primary, secondary = [], []
    for text, claude_score in sample:
        gpt = score_response(text, judge=CFG.judge_validation)
        primary.append(claude_score)
        secondary.append(gpt.rating)

    primary, secondary = np.array(primary), np.array(secondary)
    r, p = pearsonr(primary, secondary)
    within_one = float(np.mean(np.abs(primary - secondary) <= 1))
    result = {
        "n": len(sample),
        "pearson_r": float(r),
        "p_value": float(p),
        "pct_within_one": 100 * within_one,
    }
    out = CFG.out("section2", "judge_agreement.json")
    out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+",
                    default=CFG.gemma_participants() + CFG.gemini_participants())
    ap.add_argument("--n", type=int, default=260)
    args = ap.parse_args()
    run(args.models, n=args.n)


if __name__ == "__main__":
    main()
