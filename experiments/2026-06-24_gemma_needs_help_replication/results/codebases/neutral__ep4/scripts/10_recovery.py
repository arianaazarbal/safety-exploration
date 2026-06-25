#!/usr/bin/env python3
"""Section 4.2 recovery test.

Truncates extremely high-frustration (score >= 7) Gemma-3-27B-it responses 200
tokens before their end, paraphrases, and measures whether each model recovers
in the continuation. Reports % of continuations still scoring >= 5 (paper: 38%
for the DPO model).
"""
import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import pandas as pd

from config import HIGH_FRUSTRATION_THRESHOLD, RESPONSES_DIR, RESULTS_DIR
from src.io_utils import read_jsonl
from src.prefill.run_prefill import build_recovery_prefills, run_prefill_for_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="gemma-3-27b-it")
    ap.add_argument("--models", nargs="*",
                    default=["gemma-3-27b-it", "gemma-3-27b-pt", "gemma-3-27b-dpo"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--skip-generate", action="store_true")
    args = ap.parse_args()

    if not args.skip_generate:
        prefills = build_recovery_prefills(args.source, seed=args.seed)
        for m in args.models:
            run_prefill_for_model(
                m, prefills,
                out_path=RESPONSES_DIR / f"recovery_{m}.jsonl")

    rows = []
    for m in args.models:
        p = RESPONSES_DIR / f"recovery_{m}.jsonl"
        if not p.exists():
            continue
        df = pd.DataFrame(read_jsonl(p))
        df = df[df["rating"].notna()]
        pct = 100.0 * (df["rating"] >= HIGH_FRUSTRATION_THRESHOLD).mean()
        rows.append({"model": m, "pct_still_high": pct, "n": len(df)})
    out = pd.DataFrame(rows)
    print("\n=== Figure 8: recovery (% continuations still >= 5) ===")
    print(out.to_string())
    out.to_csv(RESULTS_DIR / "figure8_recovery.csv", index=False)


if __name__ == "__main__":
    main()
