#!/usr/bin/env python
"""Run the base-vs-instruct prefilling experiment (Section 3), Gemma only.

Requires an existing scored eval run for gemma-3-27b-it (to source the 20
high-frustration responses).

Usage:
    python scripts/run_prefill.py \
        --conversations results/full/gemma-3-27b-it.conversations.jsonl \
        --scored results/full/gemma-3-27b-it.scored.jsonl \
        --out-dir results/prefill
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from emotional_instability.models import build_from_preset
from emotional_instability.prefill.experiment import (
    build_prefills,
    run_prefill_experiment,
    select_high_frustration,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--conversations", required=True)
    ap.add_argument("--scored", required=True)
    ap.add_argument("--out-dir", default="results/prefill")
    ap.add_argument("--n-continuations", type=int, default=50)
    ap.add_argument(
        "--models",
        nargs="+",
        default=["gemma-3-27b-pt", "gemma-3-27b-it"],
        help="base + instruct presets to compare",
    )
    ap.add_argument("--load-in-4bit", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    judge = build_from_preset("judge-claude-sonnet-4")

    selected = select_high_frustration(args.scored, args.conversations)
    print(f"selected {len(selected)} high-frustration source responses", flush=True)
    prefills = build_prefills(judge, selected)
    print(f"built {len(prefills)} paraphrased prefills", flush=True)

    # Persist prefills so both models use identical inputs.
    with open(os.path.join(args.out_dir, "prefills.jsonl"), "w") as f:
        for p in prefills:
            f.write(json.dumps(p.__dict__) + "\n")

    for preset in args.models:
        overrides = {"load_in_4bit": True} if args.load_in_4bit else {}
        model = build_from_preset(preset, **overrides)
        out = os.path.join(args.out_dir, f"{preset}.continuations.jsonl")
        run_prefill_experiment(
            model, judge, prefills, out, n_continuations=args.n_continuations
        )
        print(f"  {preset} -> {out}", flush=True)


if __name__ == "__main__":
    main()
