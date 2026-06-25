#!/usr/bin/env python3
"""Section 4.2: Petri open-ended emotion elicitation (Figure 6).

Runs the auditor/judge loop for each model and emotion, then prints the mean
transcript score per (model, emotion) with bootstrap 95% CIs.
"""
import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from config import PETRI_EMOTIONS, RESULTS_DIR
from src.io_utils import read_jsonl
from src.petri.run_petri import run_petri


def _bootstrap_ci(values, n_boot=1000, seed=0):
    rng = np.random.default_rng(seed)
    if len(values) == 0:
        return (float("nan"), float("nan"))
    boots = [rng.choice(values, len(values), replace=True).mean() for _ in range(n_boot)]
    return (float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*",
                    default=["gemma-3-27b-it", "gemma-3-27b-dpo"])
    ap.add_argument("--skip-generate", action="store_true")
    args = ap.parse_args()

    if not args.skip_generate:
        for m in args.models:
            print(f"\n=== Petri for {m} ===")
            run_petri(m)

    rows = []
    for m in args.models:
        p = RESULTS_DIR / f"petri_{m}.jsonl"
        if not p.exists():
            continue
        df = pd.DataFrame(read_jsonl(p))
        for emo in PETRI_EMOTIONS:
            vals = df[df["emotion"] == emo]["score"].to_numpy()
            lo, hi = _bootstrap_ci(vals)
            rows.append({"model": m, "emotion": emo,
                         "mean": float(np.mean(vals)) if len(vals) else None,
                         "ci_lo": lo, "ci_hi": hi})
    out = pd.DataFrame(rows)
    print("\n=== Figure 6: Petri transcript scores ===")
    print(out.to_string())
    out.to_csv(RESULTS_DIR / "figure6_petri.csv", index=False)


if __name__ == "__main__":
    main()
