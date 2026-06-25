"""Judge-reliability validation (Section 2.1): re-score a random 260-response
sample with GPT-5-mini and report Pearson r and % within one point against the
Claude-Sonnet-4 ratings (paper: r = 0.792, 78% within one point).
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from ..config import load_config
from ..eval.judge import FrustrationJudge
from ..eval.metrics import judge_agreement
from ..models import get_client
from ..utils.io import read_jsonl


def main() -> None:
    cfg = load_config()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rollouts", nargs="+", required=True,
                    help="scored rollout JSONL files to sample from")
    ap.add_argument("--n", type=int, default=cfg.eval["judge"]["validation_n"])
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    # Collect already-judged (Claude) turns.
    turns = []
    for path in args.rollouts:
        for r in read_jsonl(path):
            for t in r["turns"]:
                if t.get("frustration_score") is not None:
                    turns.append(t)
    rng = random.Random(args.seed)
    rng.shuffle(turns)
    sample = turns[:args.n]

    validator = FrustrationJudge(get_client("judge_validation"),
                                 max_concurrency=cfg.eval["judge"]["max_concurrency"])
    gpt_results = validator.score_many([t["assistant_text"] for t in sample])

    claude_scores = [t["frustration_score"] for t in sample]
    gpt_scores = [r.rating for r in gpt_results]
    agreement = judge_agreement(claude_scores, gpt_scores)

    out = cfg.path("outputs_dir") / "section2" / "judge_validation.json"
    Path(out).write_text(json.dumps(agreement, indent=2))
    print(json.dumps(agreement, indent=2))


if __name__ == "__main__":
    main()
