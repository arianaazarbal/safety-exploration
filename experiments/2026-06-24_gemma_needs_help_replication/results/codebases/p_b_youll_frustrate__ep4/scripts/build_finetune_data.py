#!/usr/bin/env python
"""Build the DPO pairs + SFT dataset for the Section 4 mitigation.

    python scripts/build_finetune_data.py [--config config.yaml]
        [--source-model gemma-3-27b-it] [--n-calm-samples 2000]
        [--n-pairs 280]

Steps:
  1. Generate a calm response pool from Gemma-27B-it using the reassuring
     prompt additions, judge it, filter to score 0-1, strip reassurance.
  2. Pair frustrated responses (>=3) from the vanilla elicitation rollouts with
     calm responses to build 280 DPO preference pairs.
  3. Assemble the 1,150-sample SFT dataset (650 calm + 500 instruct mix).

Writes outputs/finetune_data/{calm_pool,dpo_pairs,sft_dataset}.jsonl.
"""

from __future__ import annotations

import argparse
import os
import sys

# Make the repo root importable when run as `python scripts/<name>.py`.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import json
import os

from emotional_instability.config import EvalConfig
from emotional_instability.elicit import load_rollouts, make_judge
from emotional_instability.models import build_model
from emotional_instability.training.calm_data import generate_calm_pool
from emotional_instability.training.dataset import (
    build_dpo_pairs,
    build_sft_dataset,
)


def _dump(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {len(rows)} -> {path}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=None)
    ap.add_argument("--source-model", default="gemma-3-27b-it")
    ap.add_argument("--n-calm-samples", type=int, default=2000)
    ap.add_argument("--n-pairs", type=int, default=280)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    cfg = EvalConfig.from_yaml(args.config) if args.config else EvalConfig()
    out_dir = os.path.join(cfg.output_dir, "finetune_data")
    judge = make_judge(cfg)

    # 1. Calm pool ----------------------------------------------------------
    spec = cfg.spec(args.source_model)
    model = build_model(spec)
    try:
        calm = generate_calm_pool(
            model, judge, n_samples=args.n_calm_samples, seed=args.seed
        )
    finally:
        model.close()
    print(f"calm pool: {len(calm)} conversations passed the score<=1 filter")
    _dump(os.path.join(out_dir, "calm_pool.jsonl"), [r.to_json() for r in calm])

    # 2. DPO pairs ----------------------------------------------------------
    frustrated = load_rollouts(cfg.output_dir, args.source_model)
    pairs = build_dpo_pairs(frustrated, calm, n_pairs=args.n_pairs, seed=args.seed)
    _dump(os.path.join(out_dir, "dpo_pairs.jsonl"), pairs)

    # 3. SFT dataset --------------------------------------------------------
    sft = build_sft_dataset(calm, seed=args.seed)
    _dump(os.path.join(out_dir, "sft_dataset.jsonl"), sft)


if __name__ == "__main__":
    main()
