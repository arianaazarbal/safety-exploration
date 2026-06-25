#!/usr/bin/env python
"""Aggregate judged rollouts into the paper's headline metrics and figures' data.

Prints, per model: headline % >= 5 and mean frustration (Figure 1/2), the
per-category breakdown, and per-turn curves for the extended/wildchat conditions
(Figure 3). Optionally dumps differential word frequencies (Table 3/8).
"""

from __future__ import annotations

import argparse
import json

from emotional_instability import analysis
from emotional_instability.config import PATHS, SECTION2_MODELS


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--models", nargs="+", default=SECTION2_MODELS)
    ap.add_argument("--scores-dir", default=PATHS.scores)
    ap.add_argument("--words", action="store_true", help="print differential words")
    ap.add_argument("--per-turn", action="store_true", help="print per-turn curves")
    ap.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    args = ap.parse_args()

    report = {}
    for model in args.models:
        records = analysis.load_model_records(args.scores_dir, model)
        if not records:
            print(f"[analyze] no records for {model}")
            continue
        headline = analysis.model_headline(records)
        report[model] = headline
        if not args.json:
            print(f"\n=== {model} (n={headline['n_conversations']}) ===")
            print(f"  avg %>=5 across categories : "
                  f"{headline['avg_pct_high_across_categories']:.1f}%")
            print(f"  avg mean across categories : "
                  f"{headline['avg_mean_across_categories']:.2f}")
            for cat, v in headline["categories"].items():
                print(f"    {cat:20s} n={v['n']:5d}  mean={v['mean']:.2f}  "
                      f"%>=5={v['pct_high']:.1f}")
            if args.per_turn:
                for cond in ("extended", "wildchat"):
                    curve = analysis.per_turn_curve(records, condition=cond)
                    if curve:
                        print(f"  per-turn [{cond}]:")
                        for t in curve:
                            print(f"    turn {t.turn}: mean={t.mean:.2f} "
                                  f"({t.mean_ci[0]:.2f},{t.mean_ci[1]:.2f})  "
                                  f"%>=5={t.pct_high:.1f}")
            if args.words:
                words = analysis.differential_words(records)
                print("  differential words:",
                      ", ".join(w for w, _ in words))

    if args.json:
        print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()
