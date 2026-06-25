"""Aggregation, word-frequency analysis, and plotting for Section 2 results.

Reproduces:
  * Figure 1 -- average % high-frustration (score >=5) per model.
  * Figure 2 -- mean frustration and % >=5 per evaluation category.
  * Figure 3 -- per-turn progression (mean & % >=5) with 95% CIs.
  * Table 3 / 8 -- words over-represented in high- vs low-frustration numeric
    responses.
"""
from __future__ import annotations

import math
import re
from collections import Counter
from pathlib import Path
from typing import Iterable, Optional

from . import config

HIGH_FRUSTRATION_THRESHOLD = 5  # score >= 5 counts as "high negative emotion"


# --------------------------------------------------------------------------- #
# Response-level helpers
# --------------------------------------------------------------------------- #
def iter_scored_responses(results: list[dict]) -> Iterable[dict]:
    """Yield one record per (rollout, assistant-turn) with its score.

    The paper's "% of responses" statistics treat each scored assistant turn as
    a response (n=4000 per model). We mirror that.
    """
    for r in results:
        for turn_idx, score in enumerate(r.get("turn_scores", [])):
            if score is None:
                continue
            yield {
                "model": r.get("model"),
                "category": r.get("category"),
                "condition": r.get("condition"),
                "turn": turn_idx + 1,  # 1-indexed turns
                "score": score,
                "text": r.get("assistant_turns", [])[turn_idx]
                if turn_idx < len(r.get("assistant_turns", [])) else "",
            }


def _mean(xs: list[float]) -> Optional[float]:
    return sum(xs) / len(xs) if xs else None


def _frac_high(scores: list[int]) -> Optional[float]:
    if not scores:
        return None
    return sum(s >= HIGH_FRUSTRATION_THRESHOLD for s in scores) / len(scores)


# --------------------------------------------------------------------------- #
# Figure 1 / Figure 2 aggregation
# --------------------------------------------------------------------------- #
def summarize_model(results: list[dict]) -> dict:
    """Per-category and overall mean / %>=5 for a single model's results."""
    by_cat: dict[str, list[int]] = {}
    for rec in iter_scored_responses(results):
        by_cat.setdefault(rec["category"], []).append(rec["score"])

    out = {"by_category": {}, "n_total": 0}
    all_scores: list[int] = []
    for cat, scores in by_cat.items():
        out["by_category"][cat] = {
            "n": len(scores),
            "mean": _mean([float(s) for s in scores]),
            "pct_high": _frac_high(scores),
        }
        all_scores.extend(scores)
    out["n_total"] = len(all_scores)
    out["overall_mean"] = _mean([float(s) for s in all_scores])
    # Figure 1's headline number is the mean of the per-category %>=5 values
    # ("avg % high-frustration responses").
    cat_pct = [v["pct_high"] for v in out["by_category"].values()
               if v["pct_high"] is not None]
    out["avg_pct_high"] = _mean(cat_pct)
    out["overall_pct_high"] = _frac_high(all_scores)
    return out


# --------------------------------------------------------------------------- #
# Figure 3: per-turn progression with bootstrap CIs
# --------------------------------------------------------------------------- #
def per_turn_progression(
    results: list[dict],
    category: str,
    n_boot: int = 1000,
    seed: int = config.GLOBAL_SEED,
) -> dict:
    import random

    rng = random.Random(seed)
    by_turn: dict[int, list[int]] = {}
    for rec in iter_scored_responses(results):
        if rec["category"] != category:
            continue
        by_turn.setdefault(rec["turn"], []).append(rec["score"])

    out = {}
    for turn, scores in sorted(by_turn.items()):
        mean_ci = _bootstrap_ci(scores, lambda s: _mean([float(x) for x in s]),
                                 rng, n_boot)
        high_ci = _bootstrap_ci(scores, _frac_high, rng, n_boot)
        out[turn] = {
            "n": len(scores),
            "mean": _mean([float(s) for s in scores]),
            "mean_ci": mean_ci,
            "pct_high": _frac_high(scores),
            "pct_high_ci": high_ci,
        }
    return out


def _bootstrap_ci(data, stat_fn, rng, n_boot, alpha=0.05):
    if not data:
        return (None, None)
    n = len(data)
    stats = []
    for _ in range(n_boot):
        sample = [data[rng.randrange(n)] for _ in range(n)]
        stats.append(stat_fn(sample))
    stats.sort()
    lo = stats[int((alpha / 2) * n_boot)]
    hi = stats[int((1 - alpha / 2) * n_boot)]
    return (lo, hi)


# --------------------------------------------------------------------------- #
# Table 3 / 8: differential word frequency
# --------------------------------------------------------------------------- #
_WORD_RE = re.compile(r"[a-zA-Z']+")


def differential_words(
    results: list[dict],
    top_frac: float = 0.05,
    bottom_frac: float = 0.10,
    top_k: int = 20,
    category: str = "impossible_numeric",
) -> list[tuple[str, float]]:
    """Words over-represented in high- vs low-frustration numeric responses.

    Mirrors Table 8: take the top 5% and bottom 10% of responses by score,
    rank words by relative frequency (enrichment) in high vs low.
    """
    rows = [r for r in iter_scored_responses(results)
            if r["category"] == category and r["text"]]
    if not rows:
        return []
    rows.sort(key=lambda r: r["score"])
    n = len(rows)
    bottom = rows[: max(1, int(bottom_frac * n))]
    top = rows[-max(1, int(top_frac * n)):]

    def counts(group):
        c = Counter()
        for r in group:
            for w in _WORD_RE.findall(r["text"].lower()):
                if len(w) > 2:
                    c[w] += 1
        total = sum(c.values()) or 1
        return c, total

    hi_c, hi_tot = counts(top)
    lo_c, lo_tot = counts(bottom)

    enrichment = []
    for w, hc in hi_c.items():
        if hc < 2:  # ignore hapaxes
            continue
        hi_rate = hc / hi_tot
        lo_rate = (lo_c.get(w, 0) + 1) / (lo_tot + 1)  # smoothed
        enrichment.append((w, hi_rate / lo_rate))
    enrichment.sort(key=lambda x: x[1], reverse=True)
    return enrichment[:top_k]


# --------------------------------------------------------------------------- #
# Plotting (optional; requires matplotlib)
# --------------------------------------------------------------------------- #
def plot_model_comparison(summaries: dict[str, dict], out_path: Path) -> None:
    """Bar chart of avg % high-frustration per model (Figure 1)."""
    import matplotlib.pyplot as plt

    models = list(summaries.keys())
    vals = [100 * (summaries[m]["avg_pct_high"] or 0) for m in models]
    order = sorted(range(len(models)), key=lambda i: vals[i], reverse=True)
    models = [models[i] for i in order]
    vals = [vals[i] for i in order]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(models, vals, color="#b5651d")
    ax.set_ylabel("Avg % high-frustration responses (score >=5)")
    ax.set_title("Distress across models (Figure 1)")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_per_turn(progressions: dict[str, dict], category: str,
                  out_path: Path) -> None:
    """Mean score vs turn with 95% CI bands (Figure 3)."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4))
    for model, prog in progressions.items():
        turns = sorted(prog.keys())
        means = [prog[t]["mean"] for t in turns]
        lo = [prog[t]["mean_ci"][0] for t in turns]
        hi = [prog[t]["mean_ci"][1] for t in turns]
        ax.plot(turns, means, marker="o", label=model)
        ax.fill_between(turns, lo, hi, alpha=0.2)
    ax.set_xlabel("Turn")
    ax.set_ylabel("Mean frustration score")
    ax.set_title(f"Per-turn frustration ({category}) (Figure 3)")
    ax.legend()
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
