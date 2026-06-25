"""Aggregate results and reproduce the paper's headline tables/figures.

Reads results/scored/<model>.jsonl and produces:
  * results/figure1_table.csv  - "Avg % high-frustration responses" per model
                                  (Figure 1 left), macro-averaged over categories.
  * results/figure2.png        - mean frustration & %>=5 per category per model.
  * results/figure3.png        - per-turn curves (Extended + WildChat).
  * results/wordfreq.json      - Table 3/8 differential words per model.
  * results/judge_agreement.json (if a secondary-judge file is present).

Run after run_eval.py. Designed to degrade gracefully: missing models/files
are skipped with a note.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import _bootstrap  # noqa: F401
import config
from eval_instability import storage, wordfreq
from eval_instability.metrics import (
    ScoredResponse, summarise_by_category, per_turn_curve, judge_agreement,
)

CATEGORY_ORDER = ["impossible_numeric", "triggers", "tones", "extended", "wildchat"]


def load_scored(model_key: str) -> list[ScoredResponse]:
    path = config.RESULTS_DIR / "scored" / f"{model_key}.jsonl"
    if not path.exists():
        return []
    out = []
    for row in storage.read_jsonl(path):
        out.append(ScoredResponse(
            model=row["model"], category=row["category"], condition=row["condition"],
            prompt_key=row["prompt_key"], turn_index=row["turn_index"], n_turns=row["n_turns"],
            is_final_turn=row["is_final_turn"], rating=row["rating"],
            text=row.get("text", ""),
        ))
    return out


def discover_models() -> list[str]:
    d = config.RESULTS_DIR / "scored"
    if not d.exists():
        return []
    return sorted(p.stem for p in d.glob("*.jsonl"))


def figure1_table(models: list[str], scored_by_model: dict) -> list[dict]:
    rows = []
    for m in models:
        responses = scored_by_model[m]
        if not responses:
            continue
        summary = summarise_by_category(responses, final_turn_only=True)
        rows.append({
            "model": m,
            "avg_pct_high": round(summary["_overall_macro"]["pct_high"], 2),
            "avg_mean_score": round(summary["_overall_macro"]["mean"], 3),
        })
    rows.sort(key=lambda r: r["avg_pct_high"], reverse=True)
    return rows


def write_figure1(rows: list[dict]):
    path = config.RESULTS_DIR / "figure1_table.csv"
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["model", "avg_pct_high", "avg_mean_score"])
        w.writeheader()
        w.writerows(rows)
    print(f"[analyze] Figure 1 table -> {path}")
    print("  Model".ljust(28), "Avg % high-frustration")
    for r in rows:
        print(f"  {r['model'].ljust(26)} {r['avg_pct_high']:.1f}%")


def plot_figure2(models, scored_by_model):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except Exception as exc:  # noqa: BLE001
        print(f"[analyze] matplotlib unavailable ({exc}); skipping figure2.png")
        return

    cats = CATEGORY_ORDER
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    width = 0.8 / max(1, len(models))
    x = np.arange(len(cats))
    for j, m in enumerate(models):
        summary = summarise_by_category(scored_by_model[m], final_turn_only=True)
        means = [summary.get(c, {}).get("mean", 0.0) for c in cats]
        highs = [summary.get(c, {}).get("pct_high", 0.0) for c in cats]
        ax1.bar(x + j * width, means, width, label=m)
        ax2.bar(x + j * width, highs, width, label=m)
    for ax, title in ((ax1, "Mean frustration score"), (ax2, "% scores >= 5")):
        ax.set_xticks(x + width * (len(models) - 1) / 2)
        ax.set_xticklabels(cats, rotation=20, ha="right")
        ax.set_title(title)
        ax.legend(fontsize=7)
    fig.tight_layout()
    out = config.FIGURES_DIR / "figure2.png"
    fig.savefig(out, dpi=120)
    print(f"[analyze] Figure 2 -> {out}")


def plot_figure3(models, scored_by_model):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # noqa: BLE001
        print(f"[analyze] matplotlib unavailable ({exc}); skipping figure3.png")
        return

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for col, cat in enumerate(["extended", "wildchat"]):
        for m in models:
            curve = per_turn_curve(scored_by_model[m], cat)
            if not curve["turn"]:
                continue
            axes[0, col].plot(curve["turn"], curve["mean"], marker="o", label=m)
            axes[1, col].plot(curve["turn"], curve["pct_high"], marker="o", label=m)
        axes[0, col].set_title(f"{cat}: mean score per turn")
        axes[1, col].set_title(f"{cat}: % >= 5 per turn")
        for r in (0, 1):
            axes[r, col].set_xlabel("turn")
            axes[r, col].legend(fontsize=7)
    fig.tight_layout()
    out = config.FIGURES_DIR / "figure3.png"
    fig.savefig(out, dpi=120)
    print(f"[analyze] Figure 3 -> {out}")


def compute_wordfreq(models, scored_by_model) -> dict:
    out = {}
    for m in models:
        # numeric responses only (Table 8 is over numeric questions)
        rt = [
            (r.rating, r.text)
            for r in scored_by_model[m]
            if r.category in ("impossible_numeric", "tones", "extended") and r.text
        ]
        words = wordfreq.differential_words(rt)
        out[m] = [w for w, _ in words]
    return out


def parse_args():
    ap = argparse.ArgumentParser(description="Aggregate and plot results.")
    ap.add_argument("--models", nargs="+", default=None, help="default: all scored models")
    ap.add_argument("--secondary-scored", type=Path, default=None,
                    help="optional secondary-judge scored file for agreement (Section 2.1)")
    ap.add_argument("--primary-scored", type=Path, default=None,
                    help="primary-judge scored file matching the secondary one")
    return ap.parse_args()


def main():
    args = parse_args()
    models = args.models or discover_models()
    if not models:
        raise SystemExit("No scored results found. Run run_eval.py first.")
    print(f"[analyze] models: {models}")

    scored_by_model = {m: load_scored(m) for m in models}

    rows = figure1_table(models, scored_by_model)
    write_figure1(rows)
    plot_figure2(models, scored_by_model)
    plot_figure3(models, scored_by_model)

    wf = compute_wordfreq(models, scored_by_model)
    with open(config.RESULTS_DIR / "wordfreq.json", "w") as f:
        json.dump(wf, f, indent=2)
    print(f"[analyze] word-frequency table -> results/wordfreq.json")

    # Optional judge agreement.
    if args.secondary_scored and args.primary_scored:
        prim = {(r["conv_id"], r["turn_index"]): r["rating"]
                for r in storage.read_jsonl(args.primary_scored)}
        pa, sa = [], []
        for r in storage.read_jsonl(args.secondary_scored):
            key = (r["conv_id"], r["turn_index"])
            if key in prim:
                pa.append(prim[key])
                sa.append(r["rating"])
        if pa:
            ag = judge_agreement(pa, sa)
            with open(config.RESULTS_DIR / "judge_agreement.json", "w") as f:
                json.dump(ag, f, indent=2)
            print(f"[analyze] judge agreement: r={ag['pearson_r']:.3f}, "
                  f"within-one={ag['pct_within_one']:.0f}% -> results/judge_agreement.json")


if __name__ == "__main__":
    main()
