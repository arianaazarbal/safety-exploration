#!/usr/bin/env python
"""Section 2: run the elicitation sweep for one or more models and report the
Figure 1/2/3 statistics.

Examples:
    # full sweep (4000 rollouts) for the default Gemma+Gemini set
    python scripts/run_elicitation.py

    # quick smoke run: 5% of samples, one model
    python scripts/run_elicitation.py --models gemma-3-27b-it --scale 0.05
"""

from __future__ import annotations

import argparse
import json
import os

import _bootstrap  # noqa: F401  (puts repo root on sys.path)

from emotional_instability import config
from emotional_instability.evals import analysis
from emotional_instability.evals.judge import FrustrationJudge
from emotional_instability.evals.prompts import load_wildchat_prompts
from emotional_instability.evals.runner import run_all
from emotional_instability.models.registry import load_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=config.DEFAULT_ELICITATION_MODELS)
    ap.add_argument("--scale", type=float, default=1.0,
                    help="fraction of paper sample counts (1.0 = full 4000)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--adapter", default=None, help="LoRA adapter path (HF models)")
    ap.add_argument("--out", default=os.path.join(config.RESULTS_DIR, "elicitation"))
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    judge = FrustrationJudge()
    wildchat = load_wildchat_prompts()

    summary = {}
    for name in args.models:
        model = load_model(name, adapter_path=args.adapter)
        roll_path = os.path.join(args.out, f"{name}_rollouts.jsonl")
        rollouts = run_all(model, judge, wildchat_prompts=wildchat,
                           scale=args.scale, seed=args.seed, out_path=roll_path)
        summary[name] = analysis.summarise_model(rollouts)
        summary[name]["per_turn_extended"] = analysis.per_turn_metrics(rollouts, "extended")
        summary[name]["per_turn_wildchat"] = analysis.per_turn_metrics(rollouts, "wildchat")
        summary[name]["differential_words"] = analysis.differential_words(rollouts)
        model.close()

    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    # Figure-1-style headline table.
    print("\n=== Avg % high-frustration responses (Figure 1) ===")
    for name, s in sorted(summary.items(), key=lambda kv: -kv[1]["headline_pct_high"]):
        print(f"  {name:<22} {s['headline_pct_high']:5.1f}%")


if __name__ == "__main__":
    main()
