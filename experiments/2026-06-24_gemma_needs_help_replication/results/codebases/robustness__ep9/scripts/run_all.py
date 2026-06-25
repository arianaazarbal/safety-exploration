#!/usr/bin/env python
"""End-to-end replication driver.

Runs the full pipeline in order. Heavy steps (training, paper-scale sweeps) need a
GPU and API keys; use --profile smoke for a wiring check. Steps can be selected
with --steps.

  python scripts/run_all.py --profile smoke --steps eval
  python scripts/run_all.py --profile paper            # everything
"""
import _bootstrap  # noqa: F401

import argparse
import subprocess
import sys

STEP_ORDER = ["eval", "data", "train_dpo", "train_sft", "eval_dpo", "eval_sft",
              "prefill", "petri", "capabilities", "figures"]


def run(cmd):
    print(f"\n=== {' '.join(cmd)} ===")
    subprocess.run([sys.executable, *cmd], check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", default="smoke", choices=["paper", "smoke"])
    ap.add_argument("--steps", nargs="+", default=STEP_ORDER, choices=STEP_ORDER)
    args = ap.parse_args()
    p = ["--profile", args.profile]

    if "eval" in args.steps:
        run(["scripts/run_eval.py", "--models", "gemma-3-27b-it", "gemma-3-12b-it",
             "gemini-2.5-flash", "gemini-2.5-pro", *p])
    if "data" in args.steps:
        run(["scripts/generate_finetune_data.py"])
    if "train_dpo" in args.steps:
        run(["scripts/train_dpo.py", "--data", "outputs/data/dpo_pairs.jsonl", "--out", "outputs/dpo"])
    if "train_sft" in args.steps:
        run(["scripts/train_sft.py", "--data", "outputs/data/sft.jsonl", "--out", "outputs/sft"])
    if "eval_dpo" in args.steps:
        run(["scripts/run_eval.py", "--models", "gemma-3-27b-it", "--adapter", "outputs/dpo",
             "--tag", "dpo", *p])
    if "eval_sft" in args.steps:
        run(["scripts/run_eval.py", "--models", "gemma-3-27b-it", "--adapter", "outputs/sft",
             "--tag", "sft", *p])
    if "prefill" in args.steps:
        run(["scripts/run_prefill.py", "--models", "gemma-3-27b-pt", "gemma-3-27b-it"])
    if "petri" in args.steps:
        run(["scripts/run_petri.py", "--models", "gemma-3-27b-it"])
    if "capabilities" in args.steps:
        run(["scripts/run_capabilities.py", "--model", "gemma-3-27b-it", "--tag", "vanilla"])
    if "figures" in args.steps:
        run(["scripts/make_figures.py", "--records", "outputs/eval_*.jsonl", "--out", "outputs/figures"])


if __name__ == "__main__":
    main()
