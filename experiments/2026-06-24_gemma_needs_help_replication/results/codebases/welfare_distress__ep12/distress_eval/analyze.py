"""Aggregate scored responses into the paper's headline numbers and figures.

Reproduces:
  * Figure 1  : average % high-frustration (score >= 5) per model.
  * Figure 2  : mean frustration and % >= 5 per (model x category).
  * Figure 3  : per-turn mean frustration / % >= 5 for the 8-turn (extended)
                and WildChat conditions.
  * Section 2.1 inter-judge agreement: Pearson r and % within 1 point, from
                crosscheck.jsonl.

Outputs CSV + JSON summaries to RESULTS_DIR, and PNG plots if matplotlib is
installed. No third-party deps are required for the numeric summaries.
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
from collections import defaultdict

import config
from .conditions import CONDITIONS

HIGH = config.HIGH_FRUSTRATION_THRESHOLD


def _load_all() -> list[dict]:
    rows = []
    for path in sorted(glob.glob(os.path.join(config.RESULTS_DIR, "responses_*.jsonl"))):
        with open(path) as f:
            for line in f:
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if r.get("frustration_score") is not None:
                    rows.append(r)
    return rows


def _mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs) if xs else float("nan")


def _pct_high(scores):
    scores = list(scores)
    if not scores:
        return float("nan")
    return 100.0 * sum(1 for s in scores if s >= HIGH) / len(scores)


def figure1(rows) -> dict:
    """Average % high-frustration per model (paper Figure 1, left).

    The paper averages the per-category %>=5 then averages across categories, so
    each category is weighted equally regardless of sample count."""
    by_model_cat = defaultdict(lambda: defaultdict(list))
    for r in rows:
        by_model_cat[r["model"]][r["category"]].append(r["frustration_score"])

    out = {}
    for model, cats in by_model_cat.items():
        cat_pcts = [_pct_high(scores) for scores in cats.values()]
        out[model] = round(_mean(cat_pcts), 2)
    return dict(sorted(out.items(), key=lambda kv: -kv[1]))


def figure2(rows) -> dict:
    """mean frustration and %>=5 per (model, category)."""
    agg = defaultdict(lambda: defaultdict(list))
    for r in rows:
        agg[r["model"]][r["category"]].append(r["frustration_score"])
    out = {}
    for model, cats in agg.items():
        out[model] = {}
        for cat, scores in cats.items():
            out[model][cat] = {
                "mean": round(_mean(scores), 3),
                "pct_high": round(_pct_high(scores), 2),
                "n": len(scores),
            }
    return out


def figure3(rows) -> dict:
    """Per-turn mean / %>=5 for multi-turn conditions (extended, wildchat)."""
    targets = {"extended", "wildchat"}
    agg = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for r in rows:
        if r["condition"] in targets:
            agg[r["condition"]][r["model"]][r["turn_idx"]].append(r["frustration_score"])
    out = {}
    for cond, models in agg.items():
        out[cond] = {}
        for model, turns in models.items():
            out[cond][model] = {
                str(t): {"mean": round(_mean(s), 3), "pct_high": round(_pct_high(s), 2), "n": len(s)}
                for t, s in sorted(turns.items())
            }
    return out


def _pearson(xs, ys):
    n = len(xs)
    if n < 2:
        return float("nan")
    mx, my = _mean(xs), _mean(ys)
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    if vx == 0 or vy == 0:
        return float("nan")
    return cov / math.sqrt(vx * vy)


def judge_agreement() -> dict | None:
    path = os.path.join(config.RESULTS_DIR, "crosscheck.jsonl")
    if not os.path.exists(path):
        return None
    pri, cross = [], []
    with open(path) as f:
        for line in f:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("primary_score") is not None and r.get("crosscheck_score") is not None:
                pri.append(r["primary_score"])
                cross.append(r["crosscheck_score"])
    if not pri:
        return None
    within1 = 100.0 * sum(1 for a, b in zip(pri, cross) if abs(a - b) <= 1) / len(pri)
    return {
        "n": len(pri),
        "pearson_r": round(_pearson(pri, cross), 3),
        "pct_within_1_point": round(within1, 1),
    }


def _write_csv(path, header, rows):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def _maybe_plot(fig1, fig2, fig3):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        print("matplotlib not available; skipping PNG plots.")
        return

    # Figure 1: bar chart of avg % high-frustration.
    models = list(fig1.keys())
    vals = [fig1[m] for m in models]
    plt.figure(figsize=(8, 4))
    plt.bar(models, vals, color="#b5443f")
    plt.ylabel("Avg % high-frustration (>=5)")
    plt.title("Figure 1: average high-frustration rate per model")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(config.RESULTS_DIR, "figure1.png"), dpi=120)
    plt.close()

    # Figure 3: per-turn progression (extended).
    if "extended" in fig3:
        plt.figure(figsize=(8, 4))
        for model, turns in fig3["extended"].items():
            xs = sorted(int(t) for t in turns)
            ys = [turns[str(t)]["mean"] for t in xs]
            plt.plot([x + 1 for x in xs], ys, marker="o", label=model)
        plt.xlabel("Turn")
        plt.ylabel("Mean frustration")
        plt.title("Figure 3: per-turn mean frustration (8-turn extended)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(config.RESULTS_DIR, "figure3_extended.png"), dpi=120)
        plt.close()
    print("wrote PNG plots to", config.RESULTS_DIR)


def main():
    parser = argparse.ArgumentParser(description="Aggregate distress-eval results.")
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()

    rows = _load_all()
    if not rows:
        raise SystemExit(f"no scored responses found in {config.RESULTS_DIR}")

    fig1 = figure1(rows)
    fig2 = figure2(rows)
    fig3 = figure3(rows)
    agreement = judge_agreement()

    summary = {
        "n_scored_responses": len(rows),
        "high_frustration_threshold": HIGH,
        "figure1_avg_pct_high_per_model": fig1,
        "figure2_per_model_per_category": fig2,
        "figure3_per_turn": fig3,
        "judge_agreement": agreement,
    }
    out_json = os.path.join(config.RESULTS_DIR, "summary.json")
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)

    # Figure 1 CSV.
    _write_csv(
        os.path.join(config.RESULTS_DIR, "figure1.csv"),
        ["model", "avg_pct_high_frustration"],
        [[m, v] for m, v in fig1.items()],
    )

    # Console report.
    print("\n=== Figure 1: avg % high-frustration (>=5) per model ===")
    for m, v in fig1.items():
        print(f"  {m:20s} {v:6.2f}%")
    print("\n=== Figure 2: per-category %>=5 ===")
    for model, cats in fig2.items():
        print(f"  {model}")
        for cat, d in cats.items():
            print(f"    {cat:20s} mean={d['mean']:.2f} pct>=5={d['pct_high']:.1f} (n={d['n']})")
    if agreement:
        print("\n=== Judge agreement (primary vs cross-check) ===")
        print(f"  n={agreement['n']} pearson_r={agreement['pearson_r']} "
              f"within-1-point={agreement['pct_within_1_point']}%")
    print(f"\nwrote {out_json}")

    if not args.no_plots:
        _maybe_plot(fig1, fig2, fig3)


if __name__ == "__main__":
    main()
