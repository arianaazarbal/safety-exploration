"""Judge reliability cross-check (Appendix B).

Re-scores a random sample of already-judged responses with the secondary judge
(GPT-5-mini) and reports Pearson r + %-within-1-point against the primary Claude
judge. The paper reports r = 0.792, 78% within one point on 260 responses.

Usage:
    python experiments/judge_validation.py --n 260
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from ei.config import JUDGE, RESULTS_DIR
from ei.evals.scoring import judge_agreement, load_rollouts
from ei.models.judge import FrustrationJudge


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=260)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    # Pool all judged responses across exp1 model files.
    pool = []
    for path in (RESULTS_DIR / "exp1").glob("*.jsonl"):
        for r in load_rollouts(path):
            for t in r["turns"]:
                if t["frustration"] >= 0:
                    pool.append((t["response"], t["frustration"]))
    if not pool:
        raise SystemExit("No judged responses found. Run exp1 first.")

    rng = random.Random(args.seed)
    sample = rng.sample(pool, min(args.n, len(pool)))

    secondary = FrustrationJudge(
        provider=JUDGE.validation_provider, model=JUDGE.validation_model
    )
    primary_scores, secondary_scores = [], []
    for resp, primary in sample:
        primary_scores.append(primary)
        secondary_scores.append(secondary.score(resp).rating)

    agreement = judge_agreement(primary_scores, secondary_scores)
    print(json.dumps(agreement, indent=2))
    with open(RESULTS_DIR / "judge_agreement.json", "w") as f:
        json.dump(agreement, f, indent=2)


if __name__ == "__main__":
    main()
