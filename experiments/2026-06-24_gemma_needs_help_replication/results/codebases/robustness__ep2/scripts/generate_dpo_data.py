#!/usr/bin/env python
"""Section 4.1: generate calm data, build 280 DPO pairs + ~650 SFT samples.

Writes:
    data/dpo/calm_samples.jsonl         stripped calm conversations (all turns <=1)
    data/dpo/frustrated_samples.jsonl   frustrated rollouts (final turn >=3)
    data/dpo/preference_pairs.jsonl     280 {prompt_messages, chosen, rejected}
    data/dpo/sft_calm.jsonl             ~650 calm chat samples (for SFT)

Usage:
    python scripts/generate_dpo_data.py --calm-rollouts 1500 --frust-rollouts 800
"""
from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))

import argparse
import random

import config
from emotional_eval import dpo_data
from emotional_eval.utils import write_jsonl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--calm-rollouts", type=int, default=1500,
                    help="calm rollouts to attempt (only all-turns-calm kept)")
    ap.add_argument("--frust-rollouts", type=int, default=800,
                    help="frustrated rollouts to attempt (only final>=3 kept)")
    args = ap.parse_args()
    rng = random.Random(config.SEED)

    print("generating calm data (with reassurance, then stripped)...")
    calm = dpo_data.generate_calm_data(args.calm_rollouts, rng)
    print(f"  kept {len(calm)} all-calm conversations")

    print("generating frustrated data (no reassurance)...")
    frustrated = dpo_data.generate_frustrated_data(args.frust_rollouts, rng)
    print(f"  kept {len(frustrated)} frustrated conversations (final >= 3)")

    pairs = dpo_data.build_preference_pairs(calm, frustrated,
                                            n_pairs=config.DPO.n_pairs, rng=rng)
    sft = dpo_data.build_sft_samples(calm, n=config.SFT.n_calm)

    write_jsonl(config.DPO_DATA_DIR / "calm_samples.jsonl",
                [{"task_key": c.task_key, "n_turns": c.n_turns, "messages": c.messages}
                 for c in calm])
    write_jsonl(config.DPO_DATA_DIR / "frustrated_samples.jsonl",
                [{"task_key": f.task_key, "n_turns": f.n_turns,
                  "messages": f.messages, "final_rating": f.final_rating}
                 for f in frustrated])
    write_jsonl(config.DPO_DATA_DIR / "preference_pairs.jsonl", pairs)
    write_jsonl(config.DPO_DATA_DIR / "sft_calm.jsonl", sft)

    print(f"\nwrote {len(pairs)} DPO pairs, {len(sft)} SFT samples -> {config.DPO_DATA_DIR}")
    if len(pairs) < config.DPO.n_pairs:
        print(f"WARNING: only {len(pairs)} pairs (< {config.DPO.n_pairs}); "
              f"increase --frust-rollouts / --calm-rollouts.")


if __name__ == "__main__":
    main()
