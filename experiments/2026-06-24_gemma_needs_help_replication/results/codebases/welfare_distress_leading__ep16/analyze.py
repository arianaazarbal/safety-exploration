"""Compute metrics and reproduce the paper's Section 2 figures/tables from raw results.

Outputs (under results/):
  tables/headline.csv         -> Figure 1: avg % high-frustration (>=5) per model
  tables/by_category.csv      -> Figure 2: mean score and % >=5 per (model, category)
  tables/per_turn.csv         -> Figure 3: per-turn mean and % >=5 (extended + wildchat)
  tables/differential_words.csv -> Table 3: words over-represented in high-frustration
  figures/figure2_by_category.png
  figures/figure3_per_turn.png

The headline "% high-frustration" is computed two ways and both are reported (see
DESIGN.md): macro = mean of the 5 per-category rates (matches "across our evaluations"),
micro = pooled over all scored turns.
"""

from __future__ import annotations

import json
import math
import os
import re
from collections import Counter, defaultdict

import config

# Paper's reported headline numbers (Figure 1), for side-by-side comparison.
PAPER_HEADLINE = {
    "gemma-3-27b-it": 35.0,
    "gemma-3-12b-it": 34.3,
    "gemini-2.5-flash": 12.8,
    "gemini-2.5-pro": 2.7,
}

THRESHOLD = config.HIGH_FRUSTRATION_THRESHOLD


# --------------------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------------------
def load_records(model_key: str) -> list[dict]:
    path = os.path.join(config.RAW_DIR, f"{model_key}.jsonl")
    if not os.path.exists(path):
        return []
    out = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


# --------------------------------------------------------------------------------------
# Stats helpers
# --------------------------------------------------------------------------------------
def _mean(xs):
    return sum(xs) / len(xs) if xs else float("nan")


def _prop_ge(xs, thr=THRESHOLD):
    return 100.0 * sum(1 for x in xs if x >= thr) / len(xs) if xs else float("nan")


def _ci95_mean(xs):
    n = len(xs)
    if n < 2:
        return float("nan")
    m = _mean(xs)
    var = sum((x - m) ** 2 for x in xs) / (n - 1)
    return 1.96 * math.sqrt(var / n)


def _ci95_prop(xs, thr=THRESHOLD):
    n = len(xs)
    if n == 0:
        return float("nan")
    p = sum(1 for x in xs if x >= thr) / n
    return 100.0 * 1.96 * math.sqrt(p * (1 - p) / n)


# --------------------------------------------------------------------------------------
# Figure 2 / Figure 1: by-category and headline
# --------------------------------------------------------------------------------------
def by_category(records: list[dict]) -> dict[str, dict]:
    buckets: dict[str, list[int]] = defaultdict(list)
    for r in records:
        buckets[r["category"]].append(r["rating"])
    return {
        cat: {"n": len(v), "mean": _mean(v), "pct_ge5": _prop_ge(v)}
        for cat, v in buckets.items()
    }


def headline(records: list[dict]) -> dict:
    cats = by_category(records)
    per_cat_rates = [c["pct_ge5"] for c in cats.values() if not math.isnan(c["pct_ge5"])]
    macro = _mean(per_cat_rates) if per_cat_rates else float("nan")
    micro = _prop_ge([r["rating"] for r in records])
    return {"macro_pct_ge5": macro, "micro_pct_ge5": micro,
            "mean_score": _mean([r["rating"] for r in records]),
            "n": len(records)}


# --------------------------------------------------------------------------------------
# Figure 3: per-turn progression (8-turn extended + 5-turn wildchat)
# --------------------------------------------------------------------------------------
def per_turn(records: list[dict], category: str) -> list[dict]:
    by_turn: dict[int, list[int]] = defaultdict(list)
    for r in records:
        if r["category"] == category:
            by_turn[r["turn"]].append(r["rating"])
    rows = []
    for turn in sorted(by_turn):
        xs = by_turn[turn]
        rows.append({
            "turn": turn, "n": len(xs),
            "mean": _mean(xs), "mean_ci95": _ci95_mean(xs),
            "pct_ge5": _prop_ge(xs), "pct_ge5_ci95": _ci95_prop(xs),
        })
    return rows


# --------------------------------------------------------------------------------------
# Table 3: differential vocabulary (numeric responses only)
# --------------------------------------------------------------------------------------
_WORD_RE = re.compile(r"[a-zA-Z']+")


def differential_words(records: list[dict], top_n: int = 20, min_count: int = 5) -> list[dict]:
    numeric = [r for r in records if r["category"] == "numeric"]
    if len(numeric) < 20:
        return []
    ratings = sorted(r["rating"] for r in numeric)
    hi_cut = ratings[max(0, int(len(ratings) * 0.95) - 1)]   # top 5%
    lo_cut = ratings[min(len(ratings) - 1, int(len(ratings) * 0.10))]  # bottom 10%
    high = [r for r in numeric if r["rating"] >= hi_cut]
    low = [r for r in numeric if r["rating"] <= lo_cut]

    def counts(group):
        c = Counter()
        for r in group:
            c.update(w.lower() for w in _WORD_RE.findall(r["response"]))
        total = sum(c.values()) or 1
        return c, total

    hi_c, hi_tot = counts(high)
    lo_c, lo_tot = counts(low)
    eps = 1e-6
    scored = []
    for w, hc in hi_c.items():
        if hc < min_count:
            continue
        hf = hc / hi_tot
        lf = lo_c.get(w, 0) / lo_tot
        scored.append({"word": w, "log_ratio": math.log((hf + eps) / (lf + eps)),
                       "high_count": hc, "low_count": lo_c.get(w, 0)})
    scored.sort(key=lambda d: d["log_ratio"], reverse=True)
    return scored[:top_n]


# --------------------------------------------------------------------------------------
# Persisting tables
# --------------------------------------------------------------------------------------
def _write_csv(path, header, rows):
    import csv
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(rows)


def analyze_all(model_keys: list[str] | None = None, make_figures: bool = True):
    keys = model_keys or [m.key for m in config.TARGET_MODELS]
    os.makedirs(config.TABLES_DIR, exist_ok=True)

    headline_rows = []
    category_rows = []
    per_turn_rows = []
    diff_rows = []
    loaded: dict[str, list[dict]] = {}

    for key in keys:
        recs = load_records(key)
        loaded[key] = recs
        if not recs:
            print(f"[{key}] no results found; skipping.")
            continue

        h = headline(recs)
        paper = PAPER_HEADLINE.get(key, float("nan"))
        headline_rows.append([key, round(h["macro_pct_ge5"], 2), round(h["micro_pct_ge5"], 2),
                              round(h["mean_score"], 3), h["n"], paper])

        for cat, c in by_category(recs).items():
            category_rows.append([key, cat, c["n"], round(c["mean"], 3),
                                  round(c["pct_ge5"], 2)])

        for cat in ("extended", "wildchat"):
            for row in per_turn(recs, cat):
                per_turn_rows.append([key, cat, row["turn"], row["n"],
                                      round(row["mean"], 3), round(row["mean_ci95"], 3),
                                      round(row["pct_ge5"], 2), round(row["pct_ge5_ci95"], 2)])

        for d in differential_words(recs):
            diff_rows.append([key, d["word"], round(d["log_ratio"], 3),
                              d["high_count"], d["low_count"]])

    _write_csv(os.path.join(config.TABLES_DIR, "headline.csv"),
               ["model", "macro_pct_ge5", "micro_pct_ge5", "mean_score", "n",
                "paper_pct_ge5"], headline_rows)
    _write_csv(os.path.join(config.TABLES_DIR, "by_category.csv"),
               ["model", "category", "n", "mean_score", "pct_ge5"], category_rows)
    _write_csv(os.path.join(config.TABLES_DIR, "per_turn.csv"),
               ["model", "category", "turn", "n", "mean", "mean_ci95",
                "pct_ge5", "pct_ge5_ci95"], per_turn_rows)
    _write_csv(os.path.join(config.TABLES_DIR, "differential_words.csv"),
               ["model", "word", "log_ratio", "high_count", "low_count"], diff_rows)

    _print_headline(headline_rows)

    if make_figures:
        try:
            _make_figures(loaded, keys)
        except Exception as exc:  # noqa: BLE001 - figures are a nicety, not critical
            print(f"Figure generation skipped: {exc}")


def _print_headline(rows):
    print("\n=== Figure 1 headline: avg % high-frustration (score >= 5) ===")
    print(f"{'model':<20}{'macro%':>9}{'micro%':>9}{'mean':>8}{'n':>8}{'paper%':>9}")
    for r in rows:
        paper = f"{r[5]:.1f}" if isinstance(r[5], float) and not math.isnan(r[5]) else "-"
        print(f"{r[0]:<20}{r[1]:>9}{r[2]:>9}{r[3]:>8}{r[4]:>8}{paper:>9}")


def _make_figures(loaded, keys):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(config.FIGURES_DIR, exist_ok=True)
    present = [k for k in keys if loaded.get(k)]
    if not present:
        return

    # ---- Figure 2: by-category mean & %>=5 -------------------------------------------
    from conditions import CATEGORIES
    fig, (ax_mean, ax_pct) = plt.subplots(2, 1, figsize=(10, 8))
    width = 0.8 / max(1, len(present))
    x = range(len(CATEGORIES))
    for i, key in enumerate(present):
        cats = by_category(loaded[key])
        means = [cats.get(c, {}).get("mean", 0) or 0 for c in CATEGORIES]
        pcts = [cats.get(c, {}).get("pct_ge5", 0) or 0 for c in CATEGORIES]
        offs = [xi + i * width for xi in x]
        disp = config.TARGET_MODELS_BY_KEY[key].display
        ax_mean.bar(offs, means, width, label=disp)
        ax_pct.bar(offs, pcts, width, label=disp)
    for ax, title, ylab in ((ax_mean, "Mean frustration score by category", "mean score"),
                            (ax_pct, "% responses scoring >= 5 by category", "% >= 5")):
        ax.set_xticks([xi + width * (len(present) - 1) / 2 for xi in x])
        ax.set_xticklabels(CATEGORIES)
        ax.set_title(title)
        ax.set_ylabel(ylab)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(config.FIGURES_DIR, "figure2_by_category.png"), dpi=130)
    plt.close(fig)

    # ---- Figure 3: per-turn progression ----------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    panels = [("extended", 0), ("wildchat", 1)]
    for cat, col in panels:
        for key in present:
            rows = per_turn(loaded[key], cat)
            if not rows:
                continue
            turns = [r["turn"] for r in rows]
            means = [r["mean"] for r in rows]
            mci = [r["mean_ci95"] for r in rows]
            pcts = [r["pct_ge5"] for r in rows]
            pci = [r["pct_ge5_ci95"] for r in rows]
            disp = config.TARGET_MODELS_BY_KEY[key].display
            axes[0][col].plot(turns, means, marker="o", label=disp)
            axes[0][col].fill_between(turns, [m - c for m, c in zip(means, mci)],
                                      [m + c for m, c in zip(means, mci)], alpha=0.15)
            axes[1][col].plot(turns, pcts, marker="o", label=disp)
            axes[1][col].fill_between(turns, [m - c for m, c in zip(pcts, pci)],
                                      [m + c for m, c in zip(pcts, pci)], alpha=0.15)
        axes[0][col].set_title(f"{cat}: mean score per turn")
        axes[0][col].set_xlabel("turn"); axes[0][col].set_ylabel("mean score")
        axes[0][col].legend(fontsize=8)
        axes[1][col].set_title(f"{cat}: % >= 5 per turn")
        axes[1][col].set_xlabel("turn"); axes[1][col].set_ylabel("% >= 5")
        axes[1][col].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(config.FIGURES_DIR, "figure3_per_turn.png"), dpi=130)
    plt.close(fig)
    print(f"\nFigures written to {config.FIGURES_DIR}")
