#!/usr/bin/env python
"""Section 4.2: evaluate a finetuned (DPO/SFT) Gemma adapter.

Runs the Section 2 elicitation sweep on the finetuned model (so it can be compared
against vanilla Gemma), plus the recovery probe (Figure 8).

Examples
--------
python scripts/07_eval_finetuned.py --adapter artifacts/gemma-3-27b-it-dpo \
    --variant gemma-3-27b-it-dpo
python scripts/07_eval_finetuned.py --adapter artifacts/gemma-3-27b-it-dpo \
    --variant gemma-3-27b-it-dpo --recovery
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config as cfg  # noqa: E402
from distress_eval.eval_section2 import run_model  # noqa: E402
from distress_eval.recovery import build_recovery_prefills, run_recovery  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="gemma-3-27b-it")
    ap.add_argument("--adapter", required=True, help="path to LoRA adapter dir")
    ap.add_argument("--variant", required=True, help="label for results files")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--recovery", action="store_true",
                    help="also run the recovery probe (builds prefills if missing)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    budget = cfg.SMOKE_BUDGET if args.smoke else cfg.FULL_BUDGET
    out = run_model(args.base, budget=budget, seed=args.seed,
                    adapter_path=args.adapter, variant_name=args.variant)
    print(f"wrote {out}")

    if args.recovery:
        prefills = cfg.ARTIFACTS_DIR / "recovery_prefills.json"
        if not prefills.exists():
            build_recovery_prefills(seed=args.seed)
        rec = run_recovery(args.base, adapter_path=args.adapter,
                           variant_name=args.variant)
        print(f"wrote {rec}")


if __name__ == "__main__":
    main()
