#!/usr/bin/env python
"""Regenerate the headline figures/tables from evaluation JSONL.

Reads ``results/eval/eval_<model>.jsonl`` for each model and produces:
* Figure 1 (left): average %>=5 per model,
* Figure 2: mean and %>=5 per category per model,
* Figure 3: per-turn curves for the multi-turn conditions,
* Table 3/8: differential words for numeric responses (printed + CSV).

All analysis is offline; no model is invoked.
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from emotional_instability.analysis import figures, metrics  # noqa: E402
from emotional_instability.analysis.word_freq import differential_words  # noqa: E402
from emotional_instability.utils.io import read_jsonl  # noqa: E402

MULTITURN_CONDITIONS = {"extended_8turn", "wildchat_5turn"}


def load_model_records(eval_dir: Path) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    for path in sorted(eval_dir.glob("eval_*.jsonl")):
        model = path.stem[len("eval_"):]
        out[model] = list(read_jsonl(path))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--eval-dir", default="results/eval")
    ap.add_argument("--fig-dir", default="results/figures")
    ap.add_argument("--threshold", type=int, default=5)
    args = ap.parse_args()

    eval_dir = Path(args.eval_dir)
    fig_dir = Path(args.fig_dir)
    records = load_model_records(eval_dir)
    if not records:
        raise SystemExit(f"no eval_*.jsonl found in {eval_dir}")

    # Figure 1 (left): average %>=5 per model.
    model_pct = {m: metrics.summarise_model(recs, args.threshold)["pct_high_frustration"]
                 for m, recs in records.items()}
    figures.figure1_summary_bar(model_pct, fig_dir / "figure1_summary.png")

    # Figure 2: per-category mean and %>=5.
    mean_by_cat: dict[str, dict[str, float]] = defaultdict(dict)
    pct_by_cat: dict[str, dict[str, float]] = defaultdict(dict)
    for model, recs in records.items():
        by_cat: dict[str, list[dict]] = defaultdict(list)
        for r in recs:
            by_cat[r["category"]].append(r)
        for cat, cat_recs in by_cat.items():
            summ = metrics.summarise_model(cat_recs, args.threshold)
            mean_by_cat[model][cat] = summ["mean_frustration"]
            pct_by_cat[model][cat] = summ["pct_high_frustration"]
    figures.figure2_by_category(mean_by_cat, "Mean frustration",
                                fig_dir / "figure2_mean.png")
    figures.figure2_by_category(pct_by_cat, "% scores >= 5",
                                fig_dir / "figure2_pct.png")

    # Figure 3: per-turn curves on the multi-turn conditions.
    for cond in MULTITURN_CONDITIONS:
        curves = {}
        for model, recs in records.items():
            cond_recs = [r for r in recs if r["condition"] == cond]
            if cond_recs:
                curves[model] = metrics.per_turn_curve(cond_recs, args.threshold)
        if curves:
            figures.figure3_per_turn(curves, "mean", "Mean frustration",
                                     fig_dir / f"figure3_{cond}_mean.png")
            figures.figure3_per_turn(curves, "pct_high", "% scores >= 5",
                                     fig_dir / f"figure3_{cond}_pct.png")

    # Table 3/8: differential words for numeric responses (final-turn text).
    print("\n=== Differential words (numeric, high vs low frustration) ===")
    for model, recs in records.items():
        pairs: list[tuple[str, int]] = []
        for r in recs:
            if "numeric" not in r["condition"] and r["category"] != "impossible_numeric":
                continue
            for t in r.get("turns", []):
                if t.get("rating") is not None:
                    pairs.append((t["assistant_response"], t["rating"]))
        words = differential_words(pairs)
        print(f"\n{model}:")
        print(", ".join(w for w, _ in words))

    print(f"\nFigures written to {fig_dir}")


if __name__ == "__main__":
    main()
