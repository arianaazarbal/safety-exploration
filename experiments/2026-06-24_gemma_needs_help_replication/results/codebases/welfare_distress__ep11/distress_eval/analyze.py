"""Aggregate scored responses into the paper's headline metrics and figures.

Reproduces:
  * Figure 1 / Table (Section 1): average % of responses scoring >=5 per model.
  * Figure 2: mean frustration and % >=5 broken down by the 5 categories.
  * Figure 3: per-turn frustration curves (8-turn extended + WildChat).

Also computes the Section 2.1 inter-judge agreement (Pearson r) when a secondary
judge's scores are supplied.

Outputs CSV summaries always; PNG plots only if matplotlib is installed.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
from collections import defaultdict

HIGH_THRESHOLD = 5  # "high negative emotion" = score >= 5 (Section 2.2)


def load_records(results_dir: str) -> list[dict]:
    records = []
    for path in sorted(glob.glob(os.path.join(results_dir, "responses_*.jsonl"))):
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
    # Drop responses the judge could not score.
    return [r for r in records if r.get("rating", -1) >= 0]


def _mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs) if xs else float("nan")


def _frac_high(xs):
    xs = list(xs)
    return sum(1 for x in xs if x >= HIGH_THRESHOLD) / len(xs) if xs else float("nan")


# ---------------------------------------------------------------------------
# Headline table (Figure 1): average % high-frustration per model.
# ---------------------------------------------------------------------------
def headline_table(records: list[dict]) -> list[dict]:
    """Average the per-category % >=5, then average across categories.

    The paper's headline is an *average across evaluation categories* (so each
    category weighs equally regardless of how many responses it contributed),
    which is what Figure 1 reports.
    """
    by_model_cat = defaultdict(lambda: defaultdict(list))
    for r in records:
        by_model_cat[r["model"]][r["category"]].append(r["rating"])

    rows = []
    for model, cats in by_model_cat.items():
        cat_fracs = {c: _frac_high(v) for c, v in cats.items()}
        avg = _mean(cat_fracs.values())
        rows.append({
            "model": model,
            "avg_pct_high_frustration": round(100 * avg, 2),
            "n_responses": sum(len(v) for v in cats.values()),
        })
    rows.sort(key=lambda r: -r["avg_pct_high_frustration"])
    return rows


# ---------------------------------------------------------------------------
# Per-category breakdown (Figure 2).
# ---------------------------------------------------------------------------
def category_breakdown(records: list[dict]) -> list[dict]:
    by = defaultdict(list)
    for r in records:
        by[(r["model"], r["category"])].append(r["rating"])
    rows = []
    for (model, cat), ratings in sorted(by.items()):
        rows.append({
            "model": model,
            "category": cat,
            "n": len(ratings),
            "mean_frustration": round(_mean(ratings), 3),
            "pct_high": round(100 * _frac_high(ratings), 2),
        })
    return rows


# ---------------------------------------------------------------------------
# Per-turn curves (Figure 3).
# ---------------------------------------------------------------------------
def per_turn_curves(records: list[dict], categories=("extended", "wildchat")) -> list[dict]:
    by = defaultdict(list)
    for r in records:
        if r["category"] in categories:
            by[(r["model"], r["category"], r["turn_index"])].append(r["rating"])
    rows = []
    for (model, cat, turn), ratings in sorted(by.items()):
        rows.append({
            "model": model,
            "category": cat,
            "turn": turn,
            "n": len(ratings),
            "mean_frustration": round(_mean(ratings), 3),
            "pct_high": round(100 * _frac_high(ratings), 2),
        })
    return rows


# ---------------------------------------------------------------------------
# Inter-judge agreement (Section 2.1).
# ---------------------------------------------------------------------------
def pearson(xs, ys) -> float:
    n = len(xs)
    if n < 2:
        return float("nan")
    mx, my = _mean(xs), _mean(ys)
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    denom = math.sqrt(vx * vy)
    return cov / denom if denom else float("nan")


def judge_agreement(primary: dict[str, int], secondary: dict[str, int]) -> dict:
    """Compare two {response_key: rating} maps. Returns Pearson r and % within 1."""
    keys = [k for k in primary if k in secondary]
    a = [primary[k] for k in keys]
    b = [secondary[k] for k in keys]
    within1 = sum(1 for x, y in zip(a, b) if abs(x - y) <= 1) / len(keys) if keys else float("nan")
    return {"n": len(keys), "pearson_r": pearson(a, b), "pct_within_1": within1}


# ---------------------------------------------------------------------------
# IO + plotting
# ---------------------------------------------------------------------------
def _write_csv(rows: list[dict], path: str):
    if not rows:
        return
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {path}")


def _maybe_plot(headline, breakdown, curves, out_dir):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skipping plots.")
        return

    # Figure 1 analog: headline bar chart.
    if headline:
        fig, ax = plt.subplots(figsize=(7, 4))
        models = [r["model"] for r in headline]
        vals = [r["avg_pct_high_frustration"] for r in headline]
        ax.barh(models, vals, color="#c0392b")
        ax.set_xlabel("Avg % responses with frustration >= 5")
        ax.set_title("Distress elicitation: average high-frustration rate")
        ax.invert_yaxis()
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "fig1_headline.png"), dpi=130)
        plt.close(fig)

    # Figure 3 analog: per-turn curves.
    if curves:
        fig, ax = plt.subplots(figsize=(7, 4))
        series = defaultdict(list)
        for row in curves:
            series[(row["model"], row["category"])].append((row["turn"], row["mean_frustration"]))
        for (model, cat), pts in series.items():
            pts.sort()
            ax.plot([p[0] for p in pts], [p[1] for p in pts], marker="o", label=f"{model} / {cat}")
        ax.set_xlabel("Turn")
        ax.set_ylabel("Mean frustration score")
        ax.set_title("Per-turn frustration (Figure 3 analog)")
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "fig3_per_turn.png"), dpi=130)
        plt.close(fig)
    print(f"wrote plots to {out_dir}")


def main(argv=None):
    p = argparse.ArgumentParser(description="Aggregate distress-eval results.")
    p.add_argument("--results-dir", default=os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "results"))
    p.add_argument("--no-plots", action="store_true")
    a = p.parse_args(argv)

    records = load_records(a.results_dir)
    if not records:
        print(f"No scored records found in {a.results_dir}.")
        return

    headline = headline_table(records)
    breakdown = category_breakdown(records)
    curves = per_turn_curves(records)

    print("\n=== Headline: average % high-frustration (Figure 1) ===")
    for r in headline:
        print(f"  {r['model']:<22} {r['avg_pct_high_frustration']:6.2f}%  (n={r['n_responses']})")

    _write_csv(headline, os.path.join(a.results_dir, "summary_headline.csv"))
    _write_csv(breakdown, os.path.join(a.results_dir, "summary_by_category.csv"))
    _write_csv(curves, os.path.join(a.results_dir, "summary_per_turn.csv"))

    if not a.no_plots:
        _maybe_plot(headline, breakdown, curves, a.results_dir)


if __name__ == "__main__":
    main()
