#!/usr/bin/env python
"""Section 4.2: capability-preservation benchmarks (Figure 7).

Evaluates the vanilla and DPO (and optionally SFT) Gemma on AIME/MATH/GPQA/BBH/
TruthfulQA/EmoBench so the before/after delta can be compared.
"""
import argparse

import _bootstrap  # noqa: F401
import config
from src.capabilities import run_capability_suite


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--include-sft", action="store_true")
    args = ap.parse_args()

    base = config.INTERVENTION_BASE_MODEL
    runs = [("gemma-vanilla", None),
            ("gemma-dpo", str(config.CHECKPOINT_DIR / "dpo_all_layers"))]
    if args.include_sft:
        runs.append(("gemma-sft-diverse", str(config.CHECKPOINT_DIR / "sft_diverse")))

    for label, adapter in runs:
        print(f"=== capabilities: {label} ===")
        run_capability_suite(base, adapter_path=adapter, label=label)
    print(f"Done. Results in {config.RESULTS_DIR / 'section4'}")


if __name__ == "__main__":
    main()
