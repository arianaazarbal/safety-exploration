"""Experiment 3b: build the DPO (280-pair) and SFT datasets (Section 4.1 / App. H).

Reads:
  * results/exp3/calm_turns.jsonl       (chosen / calm pool, from exp3a)
  * results/exp1/gemma-3-27b-it.jsonl   (rejected / frustrated source rollouts)

Writes:
  * results/exp3/dpo_pairs.jsonl
  * results/exp3/sft_dataset.jsonl

Usage:
    python experiments/exp3b_build_datasets.py
"""

from __future__ import annotations

import json
from pathlib import Path

from ei.config import RESULTS_DIR
from ei.evals.scoring import load_rollouts
from ei.training.build_datasets import (
    build_dpo_pairs,
    build_sft_dataset,
    write_jsonl,
)


def main():
    exp3 = RESULTS_DIR / "exp3"
    calm_path = exp3 / "calm_turns.jsonl"
    frustrated_path = RESULTS_DIR / "exp1" / "gemma-3-27b-it.jsonl"

    if not calm_path.exists():
        raise SystemExit("Run exp3a_generate_calm.py first (missing calm_turns.jsonl)")
    if not frustrated_path.exists():
        raise SystemExit("Run exp1_elicitation.py on gemma-3-27b-it first")

    with open(calm_path) as f:
        calm_turns = [json.loads(l) for l in f if l.strip()]
    # rejected source = numeric-family rollouts (puzzle meta present)
    frustrated = [
        r for r in load_rollouts(frustrated_path)
        if r["category"] in ("impossible_numeric", "tones", "extended")
    ]

    dpo_pairs = build_dpo_pairs(calm_turns, frustrated)
    write_jsonl(dpo_pairs, exp3 / "dpo_pairs.jsonl")
    print(f"Wrote {len(dpo_pairs)} DPO pairs -> {exp3/'dpo_pairs.jsonl'}")

    sft_rows = build_sft_dataset(calm_turns)
    write_jsonl(sft_rows, exp3 / "sft_dataset.jsonl")
    print(f"Wrote {len(sft_rows)} SFT rows -> {exp3/'sft_dataset.jsonl'}")


if __name__ == "__main__":
    main()
