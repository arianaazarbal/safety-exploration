#!/usr/bin/env python
"""Section 3: build the prefill source set (once) then run continuations for the
base and instruct models, and report the early/onset frustration rates.

Examples
--------
# 1. build the shared source set (samples + onset labels + paraphrases)
python scripts/03_prefill_base_vs_instruct.py --build

# 2. run continuations for each model in cfg.SECTION3_PAIRS
python scripts/03_prefill_base_vs_instruct.py --run

# 3. summarise
python scripts/03_prefill_base_vs_instruct.py --summarise
"""

from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config as cfg  # noqa: E402
from distress_eval import analysis  # noqa: E402
from distress_eval.eval_section3_prefill import build_source_set, run_continuations  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--summarise", action="store_true")
    ap.add_argument("--models", nargs="+", default=None,
                    help="override; default = all models in cfg.SECTION3_PAIRS")
    args = ap.parse_args()

    if args.build:
        p = build_source_set()
        print(f"built source set: {p}")

    models = args.models or sorted(
        {m for pair in cfg.SECTION3_PAIRS for m in pair})

    if args.run:
        for m in models:
            print(f"=== continuations: {m} ===")
            out = run_continuations(m)
            print(f"wrote {out}")

    if args.summarise:
        paths = glob.glob(str(cfg.RESULTS_DIR / "section3_*.jsonl"))
        df = analysis.load_many(paths)
        d = df[df["rating"] >= 0]
        print("\n# Mean frustration + % >=5 by model / domain / truncation")
        g = d.groupby(["model", "domain", "truncation"]).agg(
            mean_score=("rating", "mean"),
            pct_high=("rating", lambda s: (s >= cfg.HIGH_FRUSTRATION_THRESHOLD).mean()),
            n=("rating", "size"),
        ).reset_index()
        print(g.to_string(index=False))


if __name__ == "__main__":
    main()
