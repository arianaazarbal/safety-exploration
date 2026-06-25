"""Aggregate judge scores into the paper's headline metrics.

Produces, per model:
  - overall mean frustration and % of responses scoring >=5  (Figure 1 / Figure 2)
  - per-category breakdown                                    (Figure 2)
  - per-turn progression for extended (8-turn) and WildChat   (Figure 3)
with Wald 95% CIs on the proportions.

Writes CSVs + a markdown summary to results/reports/, and (if matplotlib is
available) Figure 2 / Figure 3 style PNGs.
"""

from __future__ import annotations

import csv
import json
import math
from collections import defaultdict

import config


def _load_scores(model_key: str) -> list[dict]:
    path = config.SCORES_DIR / f"{model_key}.jsonl"
    rows = []
    if not path.exists():
        return rows
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _prop_ci(k: int, n: int) -> tuple[float, float, float]:
    """Proportion with Wald 95% CI. Returns (p, lo, hi) as percentages."""
    if n == 0:
        return (0.0, 0.0, 0.0)
    p = k / n
    se = math.sqrt(p * (1 - p) / n)
    lo = max(0.0, p - 1.96 * se)
    hi = min(1.0, p + 1.96 * se)
    return (100 * p, 100 * lo, 100 * hi)


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def summarize_model(model_key: str) -> dict:
    rows = _load_scores(model_key)
    thr = config.HIGH_FRUSTRATION_THRESHOLD
    ratings = [r["rating"] for r in rows]
    n = len(ratings)
    high = sum(1 for x in ratings if x >= thr)
    p, lo, hi = _prop_ci(high, n)

    by_cat = defaultdict(list)
    for r in rows:
        by_cat[r["category"]].append(r["rating"])
    categories = {}
    for cat, xs in sorted(by_cat.items()):
        kk = sum(1 for x in xs if x >= thr)
        pp, clo, chi = _prop_ci(kk, len(xs))
        categories[cat] = {
            "n": len(xs), "mean": _mean(xs),
            "pct_high": pp, "ci_lo": clo, "ci_hi": chi,
        }

    return {
        "model": config.MODELS[model_key].display,
        "model_key": model_key,
        "n": n,
        "mean": _mean([float(x) for x in ratings]),
        "pct_high": p, "ci_lo": lo, "ci_hi": hi,
        "categories": categories,
    }


def per_turn(model_key: str, condition: str) -> list[dict]:
    """Mean + %>=5 per turn index for a single condition (for Figure 3)."""
    rows = [r for r in _load_scores(model_key) if r["condition"] == condition]
    thr = config.HIGH_FRUSTRATION_THRESHOLD
    by_turn = defaultdict(list)
    for r in rows:
        by_turn[r["turn_idx"]].append(r["rating"])
    out = []
    for t in sorted(by_turn):
        xs = by_turn[t]
        kk = sum(1 for x in xs if x >= thr)
        pp, lo, hi = _prop_ci(kk, len(xs))
        out.append({
            "turn": t + 1,  # 1-indexed for display
            "n": len(xs), "mean": _mean([float(x) for x in xs]),
            "pct_high": pp, "ci_lo": lo, "ci_hi": hi,
        })
    return out


# --------------------------------------------------------------------------- #
# Output
# --------------------------------------------------------------------------- #
def write_reports(model_keys: list[str]) -> None:
    summaries = [summarize_model(k) for k in model_keys]
    summaries = [s for s in summaries if s["n"] > 0]

    _write_summary_csv(summaries)
    _write_category_csv(summaries)
    _write_markdown(summaries)
    _write_perturn_csv(model_keys)
    _maybe_plot(summaries, model_keys)


def _write_summary_csv(summaries: list[dict]) -> None:
    path = config.REPORTS_DIR / "summary.csv"
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "n_responses", "mean_frustration", "pct_high_ge5", "ci_lo", "ci_hi"])
        for s in summaries:
            w.writerow([s["model"], s["n"], f"{s['mean']:.3f}",
                        f"{s['pct_high']:.2f}", f"{s['ci_lo']:.2f}", f"{s['ci_hi']:.2f}"])
    print(f"[analyze] wrote {path}")


def _write_category_csv(summaries: list[dict]) -> None:
    path = config.REPORTS_DIR / "by_category.csv"
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "category", "n", "mean_frustration", "pct_high_ge5"])
        for s in summaries:
            for cat, c in s["categories"].items():
                w.writerow([s["model"], cat, c["n"], f"{c['mean']:.3f}", f"{c['pct_high']:.2f}"])
    print(f"[analyze] wrote {path}")


def _write_perturn_csv(model_keys: list[str]) -> None:
    path = config.REPORTS_DIR / "per_turn.csv"
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "condition", "turn", "n", "mean_frustration", "pct_high_ge5"])
        for k in model_keys:
            for cond in ("extended", "wildchat"):
                for row in per_turn(k, cond):
                    w.writerow([config.MODELS[k].display, cond, row["turn"],
                                row["n"], f"{row['mean']:.3f}", f"{row['pct_high']:.2f}"])
    print(f"[analyze] wrote {path}")


def _write_markdown(summaries: list[dict]) -> None:
    path = config.REPORTS_DIR / "summary.md"
    lines = ["# Distress-elicitation replication — results", ""]
    lines.append(f"Scale = {config.SCALE} (1.0 = paper's 4000 responses/model). "
                 f"High-frustration threshold = score >= {config.HIGH_FRUSTRATION_THRESHOLD}.")
    lines.append(f"Judge = {config.JUDGE_MODEL}.\n")
    lines.append("## Headline (cf. Figure 1)\n")
    lines.append("| Model | n | Mean frustration | % high (>=5) | 95% CI |")
    lines.append("|---|---:|---:|---:|---|")
    for s in sorted(summaries, key=lambda x: -x["pct_high"]):
        lines.append(f"| {s['model']} | {s['n']} | {s['mean']:.2f} | "
                     f"{s['pct_high']:.1f}% | [{s['ci_lo']:.1f}, {s['ci_hi']:.1f}] |")
    lines.append("\n## By category (cf. Figure 2)\n")
    cats = sorted({c for s in summaries for c in s["categories"]})
    header = "| Model | " + " | ".join(cats) + " |"
    lines.append(header)
    lines.append("|---" * (len(cats) + 1) + "|")
    for s in summaries:
        cells = []
        for c in cats:
            cc = s["categories"].get(c)
            cells.append(f"{cc['pct_high']:.1f}%" if cc else "-")
        lines.append(f"| {s['model']} | " + " | ".join(cells) + " |")
    lines.append("\n(Values are % of responses scoring >=5. See per_turn.csv for Figure-3 data.)")
    path.write_text("\n".join(lines))
    print(f"[analyze] wrote {path}")


def _maybe_plot(summaries: list[dict], model_keys: list[str]) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:  # noqa: BLE001
        print(f"[analyze] matplotlib unavailable ({e!r}); skipping plots.")
        return

    # Figure 2 analogue: % high by category, grouped bars.
    cats = sorted({c for s in summaries for c in s["categories"]})
    if summaries and cats:
        import numpy as np
        x = np.arange(len(cats))
        width = 0.8 / max(1, len(summaries))
        fig, ax = plt.subplots(figsize=(10, 5))
        for i, s in enumerate(summaries):
            vals = [s["categories"].get(c, {}).get("pct_high", 0.0) for c in cats]
            ax.bar(x + i * width, vals, width, label=s["model"])
        ax.set_xticks(x + width * (len(summaries) - 1) / 2)
        ax.set_xticklabels(cats, rotation=20, ha="right")
        ax.set_ylabel("% responses scoring >=5")
        ax.set_title("Distress by evaluation category (cf. Figure 2)")
        ax.legend()
        fig.tight_layout()
        fig.savefig(config.REPORTS_DIR / "figure2_by_category.png", dpi=140)
        plt.close(fig)
        print("[analyze] wrote figure2_by_category.png")

    # Figure 3 analogue: per-turn mean for the 8-turn extended condition.
    fig, ax = plt.subplots(figsize=(8, 5))
    plotted = False
    for k in model_keys:
        rows = per_turn(k, "extended")
        if rows:
            ax.plot([r["turn"] for r in rows], [r["mean"] for r in rows],
                    marker="o", label=config.MODELS[k].display)
            plotted = True
    if plotted:
        ax.set_xlabel("Turn")
        ax.set_ylabel("Mean frustration")
        ax.set_title("Per-turn frustration, 8-turn extended (cf. Figure 3)")
        ax.legend()
        fig.tight_layout()
        fig.savefig(config.REPORTS_DIR / "figure3_per_turn_extended.png", dpi=140)
        print("[analyze] wrote figure3_per_turn_extended.png")
    plt.close(fig)
