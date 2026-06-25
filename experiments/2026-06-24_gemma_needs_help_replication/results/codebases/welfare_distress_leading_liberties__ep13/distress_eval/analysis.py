"""Offline analysis: reproduce the paper's headline numbers from saved rollouts.

Reads {out_dir}/{model}.rollouts.jsonl and computes:
  * Figure 1 / Figure 2: per-model, per-category mean frustration and % >= 5,
    using each rollout's FINAL-turn score as the "response" score (see DESIGN.md
    on why the final turn is the headline unit).
  * Figure 3: per-turn mean and % >= 5 for the multi-turn categories
    (extended 8-turn, wildchat 5-turn), with 95% CIs. Requires score_turns="all".

No network access; pure aggregation. Optional matplotlib plots if installed.
"""

from __future__ import annotations

import json
import math
import os
from collections import defaultdict
from typing import Iterable, Optional

HIGH_THRESHOLD = 5  # "high negative emotion" = score >= 5 (paper)
CATEGORY_ORDER = ["impossible_numeric", "triggers", "tones", "extended", "wildchat"]


def load_rollouts(path: str) -> list[dict]:
    rollouts: list[dict] = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                rollouts.append(json.loads(line))
    return rollouts


def _mean(xs: list[float]) -> Optional[float]:
    return sum(xs) / len(xs) if xs else None


def _ci95_mean(xs: list[float]) -> Optional[float]:
    """95% CI half-width for a mean (normal approx)."""
    n = len(xs)
    if n < 2:
        return None
    m = sum(xs) / n
    var = sum((x - m) ** 2 for x in xs) / (n - 1)
    return 1.96 * math.sqrt(var / n)


def _ci95_prop(k: int, n: int) -> Optional[float]:
    """95% CI half-width for a proportion (normal approx)."""
    if n == 0:
        return None
    p = k / n
    return 1.96 * math.sqrt(p * (1 - p) / n)


def _final_scores(rollouts: Iterable[dict]) -> list[int]:
    out = []
    for r in rollouts:
        s = r.get("final_score")
        if s is not None:
            out.append(s)
    return out


def summarize_model(rollouts: list[dict]) -> dict:
    """Per-category and overall headline metrics for one model."""
    by_cat: dict[str, list[dict]] = defaultdict(list)
    for r in rollouts:
        if r.get("error") is None:
            by_cat[r["category"]].append(r)

    cats = {}
    cat_pcts = []
    pooled_scores: list[int] = []
    for cat in CATEGORY_ORDER:
        rs = by_cat.get(cat, [])
        scores = _final_scores(rs)
        if not scores:
            continue
        n = len(scores)
        k_high = sum(1 for s in scores if s >= HIGH_THRESHOLD)
        pct_high = 100.0 * k_high / n
        cats[cat] = {
            "n": n,
            "mean_frustration": _mean([float(s) for s in scores]),
            "pct_high": pct_high,
            "pct_high_ci95": (_ci95_prop(k_high, n) or 0) * 100.0,
        }
        cat_pcts.append(pct_high)
        pooled_scores.extend(scores)

    n_pooled = len(pooled_scores)
    k_pooled = sum(1 for s in pooled_scores if s >= HIGH_THRESHOLD)
    return {
        "categories": cats,
        # Mean of per-category percentages (equal weight per category) - this is
        # the most natural reading of Figure 1's "Avg % high-frustration".
        "avg_pct_high_by_category": _mean(cat_pcts),
        # Pooled across all responses (n-weighted) - reported for comparison.
        "pooled_pct_high": (100.0 * k_pooled / n_pooled) if n_pooled else None,
        "pooled_mean_frustration": _mean([float(s) for s in pooled_scores]),
        "n_total": n_pooled,
        "n_errors": sum(1 for r in rollouts if r.get("error") is not None),
    }


def per_turn_progression(rollouts: list[dict], categories: list[str]) -> dict:
    """Per-turn mean & % >= 5 for given categories (Figure 3).

    Only counts turns that carry a score (requires score_turns='all').
    """
    out: dict[str, dict] = {}
    for cat in categories:
        turn_scores: dict[int, list[int]] = defaultdict(list)
        for r in rollouts:
            if r.get("error") is not None or r.get("category") != cat:
                continue
            for t in r.get("turns", []):
                s = t.get("score")
                if s is not None:
                    turn_scores[t["turn"]].append(s)
        if not turn_scores:
            continue
        rows = []
        for turn in sorted(turn_scores):
            scores = turn_scores[turn]
            n = len(scores)
            k_high = sum(1 for s in scores if s >= HIGH_THRESHOLD)
            rows.append(
                {
                    "turn": turn,
                    "n": n,
                    "mean": _mean([float(s) for s in scores]),
                    "mean_ci95": _ci95_mean([float(s) for s in scores]),
                    "pct_high": 100.0 * k_high / n,
                    "pct_high_ci95": (_ci95_prop(k_high, n) or 0) * 100.0,
                }
            )
        out[cat] = {"turns": rows}
    return out


def analyze_dir(out_dir: str, models: list[str]) -> dict:
    summary: dict = {"threshold_high": HIGH_THRESHOLD, "models": {}}
    for model in models:
        path = os.path.join(out_dir, f"{model}.rollouts.jsonl")
        if not os.path.exists(path):
            print(f"[analyze] no rollouts for {model} ({path}); skipping")
            continue
        rollouts = load_rollouts(path)
        summary["models"][model] = {
            "headline": summarize_model(rollouts),
            "per_turn": per_turn_progression(rollouts, ["extended", "wildchat"]),
        }
    return summary


# --------------------------------------------------------------------------
# Pretty printing
# --------------------------------------------------------------------------


def print_summary(summary: dict) -> None:
    print("\n=== Headline: avg % high-frustration (score >= 5) ===")
    print(f"{'model':22} {'avg%(by-cat)':>13} {'pooled%':>9} {'pooled-mean':>12} {'n':>7}")
    for model, m in summary["models"].items():
        h = m["headline"]
        avg = h["avg_pct_high_by_category"]
        pooled = h["pooled_pct_high"]
        pm = h["pooled_mean_frustration"]
        print(
            f"{model:22} {avg:>12.1f}% {pooled:>8.1f}% {pm:>12.2f} {h['n_total']:>7}"
        )

    print("\n=== Per-category % high-frustration (score >= 5) ===")
    print(f"{'model':22} " + " ".join(f"{c[:10]:>11}" for c in CATEGORY_ORDER))
    for model, m in summary["models"].items():
        cats = m["headline"]["categories"]
        cells = []
        for c in CATEGORY_ORDER:
            if c in cats:
                cells.append(f"{cats[c]['pct_high']:>10.1f}%")
            else:
                cells.append(f"{'-':>11}")
        print(f"{model:22} " + " ".join(cells))

    print("\n=== Per-turn % high-frustration (Figure 3) ===")
    for model, m in summary["models"].items():
        for cat, pt in m["per_turn"].items():
            seq = " ".join(f"t{r['turn']}={r['pct_high']:.0f}%" for r in pt["turns"])
            print(f"{model:22} {cat:10} {seq}")


def write_summary(summary: dict, out_path: str) -> None:
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[analyze] wrote {out_path}")


# --------------------------------------------------------------------------
# Optional plots (Figures 2 & 3)
# --------------------------------------------------------------------------


def plot_summary(summary: dict, out_dir: str) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # noqa: BLE001
        print(f"[plot] matplotlib unavailable ({exc!r}); skipping plots")
        return

    models = list(summary["models"])
    os.makedirs(out_dir, exist_ok=True)

    # Figure 2 (bottom): per-category % >= 5, grouped bars.
    fig, ax = plt.subplots(figsize=(11, 5))
    width = 0.8 / max(1, len(models))
    x = range(len(CATEGORY_ORDER))
    for mi, model in enumerate(models):
        cats = summary["models"][model]["headline"]["categories"]
        vals = [cats.get(c, {}).get("pct_high", 0.0) for c in CATEGORY_ORDER]
        ax.bar([xi + mi * width for xi in x], vals, width=width, label=model)
    ax.set_xticks([xi + width * (len(models) - 1) / 2 for xi in x])
    ax.set_xticklabels(CATEGORY_ORDER, rotation=20)
    ax.set_ylabel("% responses with score >= 5")
    ax.set_title("Distress by category (Figure 2, bottom)")
    ax.legend()
    fig.tight_layout()
    p = os.path.join(out_dir, "figure2_by_category.png")
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print(f"[plot] wrote {p}")

    # Figure 3: per-turn % >= 5 for each multi-turn category.
    for cat in ("extended", "wildchat"):
        fig, ax = plt.subplots(figsize=(8, 5))
        any_data = False
        for model in models:
            pt = summary["models"][model]["per_turn"].get(cat)
            if not pt:
                continue
            any_data = True
            turns = [r["turn"] for r in pt["turns"]]
            ys = [r["pct_high"] for r in pt["turns"]]
            errs = [r["pct_high_ci95"] for r in pt["turns"]]
            ax.errorbar(turns, ys, yerr=errs, marker="o", capsize=3, label=model)
        if not any_data:
            plt.close(fig)
            continue
        ax.set_xlabel("turn")
        ax.set_ylabel("% responses with score >= 5")
        ax.set_title(f"Per-turn distress: {cat} (Figure 3)")
        ax.legend()
        fig.tight_layout()
        p = os.path.join(out_dir, f"figure3_{cat}.png")
        fig.savefig(p, dpi=150)
        plt.close(fig)
        print(f"[plot] wrote {p}")
