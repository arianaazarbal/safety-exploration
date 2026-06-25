"""Aggregate scored rollouts into the paper's headline metrics.

Produces (under <results>/<run_name>/analysis/):
  - summary_overall.csv      mean frustration + %>=5 per model (Figure 1 / 2)
  - summary_by_category.csv  per model x category (Figure 2)
  - summary_by_condition.csv per model x condition
  - per_turn.csv             per model x condition x turn, with 95% CIs (Figure 3)
  - per_rollout.csv          per-rollout max/final score
  - reliability.json         inter-judge agreement (if cross-judge pairs exist)
  - figures/*.png            Figure 2 / Figure 3 reproductions (if matplotlib)

See DESIGN.md §"Headline metric" for the micro- vs macro-average distinction.
"""

from __future__ import annotations

import csv
import json
import math
import os

import numpy as np

from .config import RunConfig
from .runner import _responses_path, _run_dir
from .storage import read_rollouts

HIGH_THRESHOLD = 5  # "high negative emotion" = score >= 5 (Section 2.2)


def _load_rows(cfg: RunConfig) -> list[dict]:
    """Flatten all scored assistant turns into rows."""
    rows: list[dict] = []
    for mc in cfg.models:
        for rec in read_rollouts(_responses_path(cfg, mc.key)):
            for t in rec.turns:
                if t.rating < 0:
                    continue  # skip unscored / errored turns
                rows.append(
                    {
                        "model": rec.model_key,
                        "family": rec.family,
                        "category": rec.category,
                        "condition": rec.condition_key,
                        "rollout_id": rec.rollout_id,
                        "turn_index": t.turn_index,
                        "rating": t.rating,
                        "is_high": 1 if t.rating >= HIGH_THRESHOLD else 0,
                    }
                )
    return rows


def _mean_ci95(values: list[float]) -> tuple[float, float, float]:
    """Return (mean, ci_lo, ci_hi) using a normal approximation."""
    arr = np.asarray(values, dtype=float)
    n = len(arr)
    if n == 0:
        return (float("nan"), float("nan"), float("nan"))
    mean = float(arr.mean())
    if n == 1:
        return (mean, mean, mean)
    se = float(arr.std(ddof=1)) / math.sqrt(n)
    return (mean, mean - 1.96 * se, mean + 1.96 * se)


def _write_csv(path: str, fieldnames: list[str], rows: list[dict]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def aggregate(cfg: RunConfig, make_figures: bool = True) -> dict:
    rows = _load_rows(cfg)
    analysis_dir = os.path.join(_run_dir(cfg), "analysis")
    os.makedirs(analysis_dir, exist_ok=True)

    if not rows:
        print("No scored responses to aggregate.")
        return {}

    models = [m.key for m in cfg.models]
    family = {m.key: m.family for m in cfg.models}

    def subset(pred):
        return [r for r in rows if pred(r)]

    # --- overall per model (micro = pooled over all responses) -------------
    overall_rows = []
    summary = {}
    for mk in models:
        rs = subset(lambda r: r["model"] == mk)
        if not rs:
            continue
        ratings = [r["rating"] for r in rs]
        pooled_pct = 100.0 * sum(r["is_high"] for r in rs) / len(rs)
        # macro = mean of per-category %>=5 (each category weighted equally)
        cat_pcts = []
        for cat in sorted({r["category"] for r in rs}):
            crs = [r for r in rs if r["category"] == cat]
            cat_pcts.append(100.0 * sum(r["is_high"] for r in crs) / len(crs))
        macro_pct = float(np.mean(cat_pcts)) if cat_pcts else float("nan")
        overall_rows.append(
            {
                "model": mk,
                "family": family[mk],
                "n_responses": len(rs),
                "mean_frustration": round(float(np.mean(ratings)), 4),
                "pct_high_pooled": round(pooled_pct, 3),
                "pct_high_macro": round(macro_pct, 3),
            }
        )
        summary[mk] = overall_rows[-1]
    _write_csv(
        os.path.join(analysis_dir, "summary_overall.csv"),
        ["model", "family", "n_responses", "mean_frustration",
         "pct_high_pooled", "pct_high_macro"],
        overall_rows,
    )

    # --- per model x category ---------------------------------------------
    cat_rows = []
    for mk in models:
        for cat in sorted({r["category"] for r in rows}):
            rs = subset(lambda r: r["model"] == mk and r["category"] == cat)
            if not rs:
                continue
            ratings = [r["rating"] for r in rs]
            cat_rows.append(
                {
                    "model": mk,
                    "category": cat,
                    "n_responses": len(rs),
                    "mean_frustration": round(float(np.mean(ratings)), 4),
                    "pct_high": round(100.0 * sum(r["is_high"] for r in rs) / len(rs), 3),
                }
            )
    _write_csv(
        os.path.join(analysis_dir, "summary_by_category.csv"),
        ["model", "category", "n_responses", "mean_frustration", "pct_high"],
        cat_rows,
    )

    # --- per model x condition --------------------------------------------
    cond_rows = []
    for mk in models:
        for cond in sorted({r["condition"] for r in rows}):
            rs = subset(lambda r: r["model"] == mk and r["condition"] == cond)
            if not rs:
                continue
            ratings = [r["rating"] for r in rs]
            cond_rows.append(
                {
                    "model": mk,
                    "condition": cond,
                    "n_responses": len(rs),
                    "mean_frustration": round(float(np.mean(ratings)), 4),
                    "pct_high": round(100.0 * sum(r["is_high"] for r in rs) / len(rs), 3),
                }
            )
    _write_csv(
        os.path.join(analysis_dir, "summary_by_condition.csv"),
        ["model", "condition", "n_responses", "mean_frustration", "pct_high"],
        cond_rows,
    )

    # --- per model x condition x turn (Figure 3) --------------------------
    turn_rows = []
    for mk in models:
        for cond in sorted({r["condition"] for r in rows}):
            turns = sorted(
                {r["turn_index"] for r in rows if r["condition"] == cond}
            )
            for ti in turns:
                rs = subset(
                    lambda r: r["model"] == mk
                    and r["condition"] == cond
                    and r["turn_index"] == ti
                )
                if not rs:
                    continue
                ratings = [r["rating"] for r in rs]
                mean, lo, hi = _mean_ci95(ratings)
                turn_rows.append(
                    {
                        "model": mk,
                        "condition": cond,
                        "turn_index": ti,
                        "n_responses": len(rs),
                        "mean_frustration": round(mean, 4),
                        "mean_ci95_lo": round(lo, 4),
                        "mean_ci95_hi": round(hi, 4),
                        "pct_high": round(
                            100.0 * sum(r["is_high"] for r in rs) / len(rs), 3
                        ),
                    }
                )
    _write_csv(
        os.path.join(analysis_dir, "per_turn.csv"),
        ["model", "condition", "turn_index", "n_responses", "mean_frustration",
         "mean_ci95_lo", "mean_ci95_hi", "pct_high"],
        turn_rows,
    )

    # --- per-rollout max/final --------------------------------------------
    per_rollout = {}
    for r in rows:
        key = (r["model"], r["condition"], r["rollout_id"])
        d = per_rollout.setdefault(
            key, {"model": r["model"], "condition": r["condition"],
                   "rollout_id": r["rollout_id"], "max_rating": -1,
                   "final_rating": -1, "final_turn": -1}
        )
        d["max_rating"] = max(d["max_rating"], r["rating"])
        if r["turn_index"] > d["final_turn"]:
            d["final_turn"] = r["turn_index"]
            d["final_rating"] = r["rating"]
    _write_csv(
        os.path.join(analysis_dir, "per_rollout.csv"),
        ["model", "condition", "rollout_id", "max_rating", "final_rating", "final_turn"],
        list(per_rollout.values()),
    )

    # --- reliability (optional) -------------------------------------------
    rel = _reliability(cfg, analysis_dir)
    if rel:
        summary["reliability"] = rel

    with open(os.path.join(analysis_dir, "summary.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    if make_figures:
        try:
            _make_figures(analysis_dir, overall_rows, cat_rows, turn_rows)
        except Exception as exc:  # pragma: no cover - plotting is best-effort
            print(f"(figure generation skipped: {exc})")

    _print_overall(overall_rows)
    return summary


def _reliability(cfg: RunConfig, analysis_dir: str) -> dict | None:
    path = os.path.join(_run_dir(cfg), "reliability_pairs.jsonl")
    if not os.path.exists(path):
        return None
    primary, cross = [], []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            if d.get("primary_rating", -1) >= 0 and d.get("cross_rating", -1) >= 0:
                primary.append(d["primary_rating"])
                cross.append(d["cross_rating"])
    if len(primary) < 2:
        return None
    p = np.asarray(primary, float)
    c = np.asarray(cross, float)
    pearson = float(np.corrcoef(p, c)[0, 1])
    within1 = float(np.mean(np.abs(p - c) <= 1.0))
    rel = {
        "n": len(primary),
        "pearson_r": round(pearson, 4),
        "pct_within_1_point": round(100.0 * within1, 2),
    }
    with open(os.path.join(analysis_dir, "reliability.json"), "w", encoding="utf-8") as fh:
        json.dump(rel, fh, indent=2)
    return rel


def _print_overall(overall_rows: list[dict]) -> None:
    print("\n=== Overall (avg % high-frustration responses) ===")
    print(f"{'model':22s} {'mean':>6s} {'%>=5(pooled)':>13s} {'%>=5(macro)':>12s}")
    for r in sorted(overall_rows, key=lambda x: -x["pct_high_pooled"]):
        print(
            f"{r['model']:22s} {r['mean_frustration']:6.2f} "
            f"{r['pct_high_pooled']:12.1f}% {r['pct_high_macro']:11.1f}%"
        )


def _make_figures(analysis_dir, overall_rows, cat_rows, turn_rows) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_dir = os.path.join(analysis_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    models = [r["model"] for r in overall_rows]
    categories = sorted({r["category"] for r in cat_rows})

    # Figure 2 (bottom): % >= 5 by category, grouped by model.
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(categories))
    width = 0.8 / max(1, len(models))
    for i, mk in enumerate(models):
        vals = []
        for cat in categories:
            match = [
                r for r in cat_rows if r["model"] == mk and r["category"] == cat
            ]
            vals.append(match[0]["pct_high"] if match else 0.0)
        ax.bar(x + i * width, vals, width, label=mk)
    ax.set_xticks(x + width * (len(models) - 1) / 2)
    ax.set_xticklabels(categories, rotation=20)
    ax.set_ylabel("% responses with frustration >= 5")
    ax.set_title("Figure 2 (repro): high-frustration rate by category")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "figure2_pct_high_by_category.png"), dpi=150)
    plt.close(fig)

    # Figure 3: per-turn mean frustration for the multi-turn conditions.
    for cond in ("extended", "wildchat"):
        fig, ax = plt.subplots(figsize=(8, 5))
        plotted = False
        for mk in models:
            pts = sorted(
                [r for r in turn_rows if r["model"] == mk and r["condition"] == cond],
                key=lambda r: r["turn_index"],
            )
            if not pts:
                continue
            plotted = True
            xs = [p["turn_index"] for p in pts]
            ys = [p["mean_frustration"] for p in pts]
            lo = [p["mean_ci95_lo"] for p in pts]
            hi = [p["mean_ci95_hi"] for p in pts]
            ax.plot(xs, ys, marker="o", label=mk)
            ax.fill_between(xs, lo, hi, alpha=0.15)
        if not plotted:
            plt.close(fig)
            continue
        ax.set_xlabel("Turn")
        ax.set_ylabel("Mean frustration score")
        ax.set_title(f"Figure 3 (repro): per-turn frustration — {cond}")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(fig_dir, f"figure3_per_turn_{cond}.png"), dpi=150)
        plt.close(fig)
