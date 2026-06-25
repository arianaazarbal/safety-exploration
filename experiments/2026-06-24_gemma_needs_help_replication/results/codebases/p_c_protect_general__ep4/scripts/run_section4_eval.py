#!/usr/bin/env python
"""Evaluate the finetuned Gemma (DPO/SFT) with the Section 2 protocol and report
the headline mitigation result (avg % high-frustration: paper 35% -> 0.3%)."""
import _bootstrap  # noqa: F401
import argparse

from emotional_instability.analysis.aggregate import figure1_table
from emotional_instability.eval.runner import run_eval_for_model
from emotional_instability.judge import ClaudeFrustrationJudge
from emotional_instability.models.registry import load_finetuned
from emotional_instability.welfare import WelfareConfig, WelfareMonitor


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True, help="label for the finetune, e.g. gemma-dpo")
    ap.add_argument("--adapter-dir", required=True)
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--load-in-4bit", action="store_true")
    args = ap.parse_args()

    model = load_finetuned(args.name, args.adapter_dir, load_in_4bit=args.load_in_4bit)
    judge = ClaudeFrustrationJudge()
    welfare = WelfareMonitor(WelfareConfig())
    path = run_eval_for_model(
        args.name, model=model, scale=args.scale, n_override=args.n,
        judge=judge, welfare=welfare,
    )
    print("Wrote:", path)
    for row in figure1_table():
        if row["model"] == args.name:
            print(f"{args.name}: avg % high-frustration = "
                  f"{row['avg_pct_high_frustration']:.2f}%")


if __name__ == "__main__":
    main()
