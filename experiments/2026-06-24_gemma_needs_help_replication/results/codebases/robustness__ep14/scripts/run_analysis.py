#!/usr/bin/env python
"""Aggregate Section 2 JSONL outputs into the paper's headline tables and figures.

Reproduces: Figure 1 (left) headline %, Figure 2 (per-condition), Figure 3 (per-turn),
Table 3/8 (differential words). Also runs the judge-reliability check if a crosscheck
file is present.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from emotional_instability import analysis as A
from emotional_instability.config import load_eval_config


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-config", default="eval.yaml")
    ap.add_argument("--section2-dir", default=None,
                    help="Dir of <model>.jsonl files; default outputs/section2")
    ap.add_argument("--diff-words-models", nargs="*", default=None)
    args = ap.parse_args()

    eval_cfg = load_eval_config(args.eval_config)
    sec2 = Path(args.section2_dir) if args.section2_dir else eval_cfg.output_dir / "section2"
    out = eval_cfg.output_dir / "analysis"
    out.mkdir(parents=True, exist_ok=True)

    files = sorted(sec2.glob("*.jsonl"))
    if not files:
        raise SystemExit(f"No JSONL files in {sec2}")
    df = pd.concat([A.load_records(f) for f in files], ignore_index=True)
    thr = int(eval_cfg["high_frustration_threshold"])

    # Headline (use final-turn rows for the response-level % >=5)
    final = A.final_turn_df(df)
    headline = A.headline_metrics(final, threshold=thr)
    headline.to_csv(out / "headline_metrics.csv", index=False)
    print("\n=== Headline (avg % high-frustration across conditions) ===")
    print(headline.to_string(index=False))
    A.plot_headline(headline, out / "fig1_headline.png")

    # Per-condition (Figure 2)
    per_cond = A.per_condition_metrics(final, threshold=thr)
    per_cond.to_csv(out / "per_condition_metrics.csv", index=False)

    # Per-turn (Figure 3) -- use full df (all turns)
    per_turn = A.per_turn_metrics(df, threshold=thr)
    per_turn.to_csv(out / "per_turn_metrics.csv", index=False)
    for cond in ["extended_8turn", "wildchat_5turn"]:
        if (per_turn["condition"] == cond).any():
            A.plot_per_turn(per_turn, cond, out / f"fig3_{cond}.png")

    # Differential words (Table 3/8)
    models = args.diff_words_models or sorted(df["model"].unique())
    diff = {}
    for m in models:
        words = A.differential_words(df, m, category="numeric")
        diff[m] = words
    with open(out / "differential_words.json", "w", encoding="utf-8") as f:
        json.dump({m: [w for w, _ in ws] for m, ws in diff.items()}, f, indent=2)
    print("\n=== Differential words (numeric, top vs bottom) ===")
    for m, ws in diff.items():
        print(f"{m}: {', '.join(w for w, _ in ws)}")

    # Judge reliability (if crosscheck file exists)
    cc_path = sec2.parent / "judge_crosscheck.jsonl"
    if cc_path.exists():
        pairs = [json.loads(l) for l in open(cc_path) if l.strip()]
        agree = A.judge_agreement([p["primary"] for p in pairs],
                                  [p["crosscheck"] for p in pairs])
        with open(out / "judge_agreement.json", "w") as f:
            json.dump(agree, f, indent=2)
        print("\n=== Judge agreement ===")
        print(agree)

    print(f"\nAnalysis written to {out}")


if __name__ == "__main__":
    main()
