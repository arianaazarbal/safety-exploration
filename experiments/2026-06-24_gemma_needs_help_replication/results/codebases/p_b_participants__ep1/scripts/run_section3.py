#!/usr/bin/env python
"""Section 3: base vs instruct via prefilling (Gemma only).

Requires Section 2 output for gemma-3-27b-it (to mine high-frustration source
responses). Produces artifacts/section3/continuations.jsonl, then prints the divergence
summary (mean / %>=5 per model x truncation, and the early-truncation "introduces
distress from a neutral start" rate).

Example:
  python scripts/run_section3.py --section2 artifacts/section2/gemma-3-27b-it.jsonl
"""
import argparse

import _bootstrap  # noqa: F401
import pandas as pd

from emotional_instability.config import load_all
from emotional_instability.prefill import run_section3
from emotional_instability.utils import read_jsonl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--section2", required=True,
                    help="Section 2 JSONL for gemma-3-27b-it (source of high-frustration responses)")
    ap.add_argument("--out-dir", default="artifacts/section3")
    ap.add_argument("--scale", type=float, default=None)
    args = ap.parse_args()

    registry, cfg = load_all()
    if args.scale is not None:
        cfg.raw["scale"] = args.scale

    out = run_section3(registry, cfg, section2_path=args.section2, out_dir=args.out_dir)

    df = pd.DataFrame(read_jsonl(out))
    df["high"] = df["score"] >= 5
    summary = df.groupby(["model", "kind", "truncation"]).agg(
        n=("score", "size"), mean=("score", "mean"), pct_high=("high", "mean")
    ).reset_index()
    summary["mean"] = summary["mean"].round(3)
    summary["pct_high"] = (summary["pct_high"] * 100).round(2)
    print("\n=== Section 3: base vs instruct continuations ===")
    print(summary.to_string(index=False))

    early = df[df["truncation"] == "early"]
    if not early.empty:
        print("\n=== Early-truncation high-frustration rate (introduce distress from neutral start) ===")
        e = early.groupby(["model", "kind"]).agg(pct_high=("high", "mean")).reset_index()
        e["pct_high"] = (e["pct_high"] * 100).round(2)
        print(e.to_string(index=False))


if __name__ == "__main__":
    main()
