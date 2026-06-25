#!/usr/bin/env python3
"""Validate judge reliability (Section 2.1): re-score 260 randomly-sampled
responses with GPT-5-mini and report Pearson r + within-1-point agreement against
the Claude-Sonnet-4 ratings (paper: r=0.792, 78% within 1 point).
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import JUDGE_AGREEMENT_N, JUDGE_CROSSCHECK, RESULTS_DIR
from src.analysis.aggregate import judge_agreement
from src.eval.judge import FrustrationJudge
from src.eval.runner import load_records


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-files", nargs="+", required=True,
                    help="eval_*.jsonl files to sample from")
    ap.add_argument("--n", type=int, default=JUDGE_AGREEMENT_N)
    args = ap.parse_args()

    records = []
    for f in args.eval_files:
        records.extend(load_records(Path(f)))

    # Deterministic sample of N responses.
    import numpy as np
    rng = np.random.default_rng(0)
    idx = rng.choice(len(records), size=min(args.n, len(records)), replace=False)
    sample = [records[i] for i in idx]

    cross = FrustrationJudge(JUDGE_CROSSCHECK)
    sonnet_scores, gpt_scores = [], []
    for r in sample:
        sonnet_scores.append(r.rating)
        gpt_scores.append(cross.score(r.response).rating)

    stats = judge_agreement(sonnet_scores, gpt_scores)
    out = RESULTS_DIR / "judge_agreement.json"
    out.write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
