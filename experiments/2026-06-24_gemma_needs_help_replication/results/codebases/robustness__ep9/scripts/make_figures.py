#!/usr/bin/env python
"""Reproduce Figures 1-3 (and the Fig 5 intervention comparison) from eval JSONL.

  python scripts/make_figures.py --records 'outputs/eval_*.jsonl' --out outputs/figures
"""
import _bootstrap  # noqa: F401

import argparse
import os

from emo_instability.analysis import (
    figure1_summary,
    figure2_by_category,
    figure3_per_turn,
    figure5_intervention,
    load_records,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", default="outputs/eval_*.jsonl",
                    help="glob of scored eval JSONL files")
    ap.add_argument("--out", default="outputs/figures")
    ap.add_argument("--turn-condition", default="extended_8turn")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    df = load_records(args.records)
    if df.empty:
        print(f"No records matched {args.records}")
        return

    f1 = figure1_summary(df, save_path=os.path.join(args.out, "figure1.png"))
    print("Figure 1 (avg % high-frustration):")
    print(f1.to_string(index=False))

    figure2_by_category(df, save_dir=args.out)
    figure3_per_turn(df, condition=args.turn_condition,
                     save_path=os.path.join(args.out, "figure3.png"))
    figure5_intervention(df, save_path=os.path.join(args.out, "figure5.png"))
    f1.to_csv(os.path.join(args.out, "figure1.csv"), index=False)
    print(f"Figures written to {args.out}")


if __name__ == "__main__":
    main()
