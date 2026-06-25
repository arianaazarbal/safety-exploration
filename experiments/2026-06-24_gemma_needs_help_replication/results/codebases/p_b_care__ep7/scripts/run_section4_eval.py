#!/usr/bin/env python3
"""Evaluate the Section 4 finetunes: re-run the distress suite, Petri, and
capability benchmarks; compare against vanilla Gemma-3-27B-it.

Requires run_section4_training.py to have produced finetune_specs.jsonl.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from gemma_needs_help import config
from gemma_needs_help.capabilities import run_capabilities
from gemma_needs_help.eval.run_eval import run_section2
from gemma_needs_help.io_utils import read_jsonl
from gemma_needs_help.petri import run_petri


def load_finetune_specs():
    path = config.CHECKPOINTS_DIR / "finetune_specs.jsonl"
    specs = {}
    for row in read_jsonl(path):
        specs[row["key"]] = config.finetune_spec(row["name"], row["adapter_path"])
    return specs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--skip-petri", action="store_true")
    ap.add_argument("--skip-capabilities", action="store_true")
    args = ap.parse_args()

    finetunes = load_finetune_specs()
    baseline = config.GEMMA_27B_IT
    eval_models = [baseline, finetunes["dpo"], finetunes["sft_diverse"], finetunes["sft_teacher"]]

    print("=== Re-running Section 2 distress suite on finetunes (Figure 5) ===")
    summary = run_section2(models=eval_models, scale=args.scale,
                           out_dir=config.RESULTS_DIR / "section4")
    print(summary["headline"].to_string(index=False))

    if not args.skip_petri:
        print("\n=== Petri open-ended elicitation (Figure 6) ===")
        petri_models = [baseline, finetunes["dpo"]]
        print(run_petri(petri_models).to_string(index=False))

    if not args.skip_capabilities:
        print("\n=== Capability preservation (Figure 7) ===")
        cap_models = [baseline, finetunes["dpo"]]
        print(run_capabilities(cap_models).to_string(index=False))


if __name__ == "__main__":
    main()
