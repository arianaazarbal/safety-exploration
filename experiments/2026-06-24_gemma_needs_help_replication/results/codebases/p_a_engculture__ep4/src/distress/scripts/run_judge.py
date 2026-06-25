"""Judge rollouts, aggregate metrics, and (optionally) run the GPT-5-mini
reliability cross-check.

Example:
    distress-judge --rollouts runs/rollouts/gemma-3-27b-it.jsonl --reliability
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from ..eval.aggregate import headline_figure1, overall_pct_high, per_category, per_turn
from ..eval.reliability import cross_check
from ..eval.runner import judge_rollouts
from ..utils import read_jsonl
from ._common import out_dir


def main() -> None:
    ap = argparse.ArgumentParser(description="Judge rollouts and aggregate.")
    ap.add_argument("--rollouts", nargs="+", required=True, help="rollout JSONL file(s)")
    ap.add_argument("--max-workers", type=int, default=8)
    ap.add_argument("--reliability", action="store_true", help="run GPT-5-mini cross-check")
    ap.add_argument("--per-turn-condition", default="extended_8turn",
                    help="condition for the per-turn figure data")
    args = ap.parse_args()

    scored_dir = out_dir("scored")
    agg_dir = out_dir("aggregates")

    all_rows: list[dict] = []
    for rp in args.rollouts:
        out = scored_dir / (Path(rp).stem + ".scored.jsonl")
        judge_rollouts(rp, out, max_workers=args.max_workers)
        all_rows.extend(read_jsonl(out))
        print(f"Scored {rp} -> {out}")

    df = pd.DataFrame(all_rows)
    per_category(df).to_csv(agg_dir / "per_category.csv", index=False)
    headline_figure1(df).to_csv(agg_dir / "headline_figure1.csv", index=False)
    overall_pct_high(df).to_csv(agg_dir / "overall.csv", index=False)
    per_turn(df).to_csv(agg_dir / "per_turn_all.csv", index=False)
    per_turn(df, condition_key=args.per_turn_condition).to_csv(
        agg_dir / f"per_turn_{args.per_turn_condition}.csv", index=False
    )
    print(f"Wrote aggregates -> {agg_dir}")

    if args.reliability:
        report = cross_check([r for r in all_rows if r.get("response")])
        (agg_dir / "reliability.json").write_text(json.dumps({
            "n": report.n, "pearson_r": report.pearson_r, "p_value": report.p_value,
            "pct_within_one": report.pct_within_one,
        }, indent=2))
        print(f"Reliability: r={report.pearson_r:.3f}, within-1={report.pct_within_one:.1f}%")


if __name__ == "__main__":
    main()
