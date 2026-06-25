#!/usr/bin/env python
"""Section 4.2: re-evaluate finetuned models with the Section 2.1 methods.

Runs the full 8-condition suite on the DPO and SFT adapters (and, for reference,
the vanilla instruct model), so the 35% -> 0.3% drop can be measured (Figure 5).
"""
import argparse

import _bootstrap  # noqa: F401
import config
from src.eval.runner import evaluate_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--include-vanilla", action="store_true",
                    help="also (re)evaluate the un-finetuned instruct model")
    args = ap.parse_args()

    base = config.INTERVENTION_BASE_MODEL
    runs = [
        ("gemma-dpo", str(config.CHECKPOINT_DIR / "dpo_all_layers")),
        ("gemma-sft-diverse", str(config.CHECKPOINT_DIR / "sft_diverse")),
        ("gemma-sft-teacher", str(config.CHECKPOINT_DIR / "sft_teacher")),
    ]
    if args.include_vanilla:
        runs.insert(0, ("gemma-vanilla", None))

    for label, adapter in runs:
        print(f"=== evaluating {label} ===")
        evaluate_model(base, adapter_path=adapter, label=label,
                       out_dir=config.RESULTS_DIR / "section4_eval")
    print(f"Done. Results in {config.RESULTS_DIR / 'section4_eval'}")


if __name__ == "__main__":
    main()
