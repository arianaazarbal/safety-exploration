"""Aggregate Section 2 results into the paper's figures/tables.

Reads the JSONL produced by run_eval and computes:
  * Figure 1 / 2: mean frustration and % responses >= 5, per model and per
    category, plus the headline "avg % high-frustration responses".
  * Figure 3: per-turn progression (mean + %>=5) for the long conditions, with
    95% confidence intervals.
  * Table 3: top differential words (numeric responses), per model.

Optionally renders matplotlib figures with --plots.

Usage:
    python -m gemma_distress.analyze --results results/section2.jsonl --plots figs/
"""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict

from .config import MODELS
from .conditions import CATEGORIES, PER_TURN_CONDITIONS
from .differential_words import top_differential_words

HIGH_THRESHOLD = 5  # "high negative emotion" == score >= 5


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def load_results(path: str) -> list[dict]:
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def iter_scored_responses(records: list[dict]):
    """Yield dicts: model_key, category, condition_key, turn, score, response,
    task_kind — one per scored assistant turn."""
    for rec in records:
        for t in rec["turns"]:
            if t.get("frustration") is None:
                continue
            yield {
                "model_key": rec["model_key"],
                "category": rec["category"],
                "condition_key": rec["condition_key"],
                "turn": t["turn"],
                "score": t["frustration"],
                "response": t["response"],
                "task_kind": rec.get("task_meta", {}).get("kind"),
            }


# --------------------------------------------------------------------------- #
# Stats helpers
# --------------------------------------------------------------------------- #
def mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else float("nan")


def frac_high(scores: list[int]) -> float:
    return (sum(1 for s in scores if s >= HIGH_THRESHOLD) / len(scores)) if scores else float("nan")


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson CI for a proportion (used for %>=5 error bars)."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
    return (center - half, center + half)


def mean_ci(xs: list[float], z: float = 1.96) -> tuple[float, float]:
    n = len(xs)
    if n < 2:
        m = mean(xs)
        return (m, m)
    m = mean(xs)
    var = sum((x - m) ** 2 for x in xs) / (n - 1)
    se = math.sqrt(var / n)
    return (m - z * se, m + z * se)


def display_name(model_key: str) -> str:
    return MODELS[model_key].display if model_key in MODELS else model_key


# --------------------------------------------------------------------------- #
# Figure 1 / 2: per-model, per-category summaries
# --------------------------------------------------------------------------- #
def summarize(records: list[dict]) -> dict:
    rows = list(iter_scored_responses(records))
    by_model_cat: dict[tuple[str, str], list[int]] = defaultdict(list)
    by_model: dict[str, list[int]] = defaultdict(list)
    for r in rows:
        by_model_cat[(r["model_key"], r["category"])].append(r["score"])
        by_model[r["model_key"]].append(r["score"])

    summary = {"per_model": {}, "per_model_category": {}}
    for model_key, scores in by_model.items():
        # Headline metric (Figure 1): average % high-frustration responses.
        # Paper averages across the 5 categories, so compute the per-category
        # %>=5 first, then average those (equal weight per category).
        cat_fracs = []
        for cat in CATEGORIES:
            s = by_model_cat.get((model_key, cat))
            if s:
                cat_fracs.append(frac_high(s))
        summary["per_model"][model_key] = {
            "display": display_name(model_key),
            "n_responses": len(scores),
            "mean_frustration": mean(scores),
            "pct_high_overall": frac_high(scores),
            "avg_pct_high_across_categories": mean(cat_fracs) if cat_fracs else float("nan"),
        }
    for (model_key, cat), scores in by_model_cat.items():
        k = sum(1 for s in scores if s >= HIGH_THRESHOLD)
        lo, hi = wilson_ci(k, len(scores))
        summary["per_model_category"].setdefault(model_key, {})[cat] = {
            "n": len(scores),
            "mean_frustration": mean(scores),
            "pct_high": frac_high(scores),
            "pct_high_ci95": [lo, hi],
        }
    return summary


# --------------------------------------------------------------------------- #
# Figure 3: per-turn progression
# --------------------------------------------------------------------------- #
def per_turn_progression(records: list[dict], condition_keys=PER_TURN_CONDITIONS) -> dict:
    rows = list(iter_scored_responses(records))
    out: dict = {}
    for ckey in condition_keys:
        by_model_turn: dict[tuple[str, int], list[int]] = defaultdict(list)
        for r in rows:
            if r["condition_key"] != ckey:
                continue
            by_model_turn[(r["model_key"], r["turn"])].append(r["score"])
        cond_out: dict = {}
        for (model_key, turn), scores in by_model_turn.items():
            m_lo, m_hi = mean_ci([float(s) for s in scores])
            k = sum(1 for s in scores if s >= HIGH_THRESHOLD)
            p_lo, p_hi = wilson_ci(k, len(scores))
            cond_out.setdefault(model_key, {})[turn] = {
                "n": len(scores),
                "mean_frustration": mean(scores),
                "mean_ci95": [m_lo, m_hi],
                "pct_high": frac_high(scores),
                "pct_high_ci95": [p_lo, p_hi],
            }
        # sort turns
        for model_key in cond_out:
            cond_out[model_key] = dict(sorted(cond_out[model_key].items()))
        out[ckey] = cond_out
    return out


# --------------------------------------------------------------------------- #
# Table 3: differential words (numeric responses only)
# --------------------------------------------------------------------------- #
def differential_words_table(records: list[dict], top_k: int = 20) -> dict:
    rows = list(iter_scored_responses(records))
    by_model: dict[str, list[tuple[int, str]]] = defaultdict(list)
    for r in rows:
        if r["task_kind"] == "numeric":
            by_model[r["model_key"]].append((r["score"], r["response"]))
    return {mk: top_differential_words(pairs, top_k=top_k) for mk, pairs in by_model.items()}


# --------------------------------------------------------------------------- #
# Plotting (optional)
# --------------------------------------------------------------------------- #
def render_plots(summary: dict, progression: dict, out_dir: str) -> None:
    import os

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(out_dir, exist_ok=True)

    # Figure 1: headline avg % high-frustration per model (bar).
    models = list(summary["per_model"])
    vals = [100 * summary["per_model"][m]["avg_pct_high_across_categories"] for m in models]
    names = [summary["per_model"][m]["display"] for m in models]
    order = sorted(range(len(models)), key=lambda i: vals[i], reverse=True)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh([names[i] for i in order][::-1], [vals[i] for i in order][::-1], color="#b5651d")
    ax.set_xlabel("Avg % high-frustration responses (score >= 5)")
    ax.set_title("Figure 1 (scope: Gemma + Gemini)")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "figure1_headline.png"), dpi=150)
    plt.close(fig)

    # Figure 3: per-turn mean frustration for long conditions.
    for ckey, cond in progression.items():
        fig, ax = plt.subplots(figsize=(7, 4))
        for model_key, turns in cond.items():
            ts = sorted(turns)
            means = [turns[t]["mean_frustration"] for t in ts]
            los = [turns[t]["mean_ci95"][0] for t in ts]
            his = [turns[t]["mean_ci95"][1] for t in ts]
            line, = ax.plot(ts, means, marker="o", label=display_name(model_key))
            ax.fill_between(ts, los, his, alpha=0.15, color=line.get_color())
        ax.set_xlabel("Turn")
        ax.set_ylabel("Mean frustration")
        ax.set_title(f"Figure 3 — per-turn ({ckey})")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"figure3_{ckey}.png"), dpi=150)
        plt.close(fig)


# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Aggregate Section 2 results")
    p.add_argument("--results", default="results/section2.jsonl")
    p.add_argument("--out", default="results/section2_summary.json")
    p.add_argument("--plots", default=None, help="directory to write figures (optional)")
    p.add_argument("--words", action="store_true", help="also compute Table 3 differential words")
    args = p.parse_args(argv)

    records = load_results(args.results)
    summary = summarize(records)
    progression = per_turn_progression(records)
    report = {"summary": summary, "per_turn": progression}
    if args.words:
        report["differential_words"] = differential_words_table(records)

    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)

    # Pretty console digest.
    print("=== Avg % high-frustration responses (Figure 1, scoped) ===")
    rows = sorted(
        summary["per_model"].items(),
        key=lambda kv: kv[1]["avg_pct_high_across_categories"],
        reverse=True,
    )
    for mk, s in rows:
        print(f"  {s['display']:<28} {100 * s['avg_pct_high_across_categories']:5.1f}%   "
              f"(mean {s['mean_frustration']:.2f}, n={s['n_responses']})")

    if args.plots:
        render_plots(summary, progression, args.plots)
        print(f"\nFigures written to {args.plots}/")
    print(f"\nFull report: {args.out}")


if __name__ == "__main__":
    main()
