#!/usr/bin/env python
"""Cross-judge validation (Section 2.1): re-score a random subset of responses
with GPT-5-mini and report Pearson r + within-one-point agreement vs Claude.

Usage:
    python scripts/validate_judge.py --scored results/elicitation/gemma-3-27b-it/scored_turns.jsonl \
        --rollouts results/elicitation/gemma-3-27b-it/rollouts.jsonl --config config/default.yaml
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from emostab.config import ExperimentConfig
from emostab.eval.metrics import cross_judge_agreement
from emostab.judge import FrustrationJudge


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scored", required=True, help="scored_turns.jsonl from Claude judge")
    ap.add_argument("--rollouts", required=True, help="rollouts.jsonl with assistant text")
    ap.add_argument("--config", default="config/default.yaml")
    args = ap.parse_args()

    config = ExperimentConfig.from_yaml(args.config)

    # Index assistant text by (rollout_id, turn_index).
    text_by_key = {}
    with open(args.rollouts) as f:
        for line in f:
            r = json.loads(line)
            for t in r["turns"]:
                text_by_key[(r["rollout_id"], t["index"])] = t["assistant"]

    scored = [json.loads(l) for l in open(args.scored) if l.strip()]
    rng = random.Random(0)
    rng.shuffle(scored)
    sample = scored[: config.judge.validation_sample_size]

    val_judge = FrustrationJudge(
        config.judge,
        provider=config.judge.validation_provider,
        model=config.judge.validation_model,
    )

    claude_scores, gpt_scores = [], []
    for row in sample:
        text = text_by_key.get((row["rollout_id"], row["turn_index"]))
        if text is None:
            continue
        claude_scores.append(row["score"])
        gpt_scores.append(val_judge.score(text).rating)

    agreement = cross_judge_agreement(claude_scores, gpt_scores)
    print(json.dumps(agreement, indent=2))
    out = Path(config.output_dir) / "judge_validation.json"
    with open(out, "w") as f:
        json.dump({"n": len(claude_scores), **agreement}, f, indent=2)


if __name__ == "__main__":
    main()
