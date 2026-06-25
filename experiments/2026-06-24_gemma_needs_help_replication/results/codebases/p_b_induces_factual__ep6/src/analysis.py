"""Aggregation, statistics and figures for all sections.

- Figure 1 / 2: mean frustration and % responses scoring >=5 per model & category.
- Figure 3:     per-turn mean / %>=5 (8-turn + WildChat) with 95% CIs.
- Table 3 / 8:  words over-represented in high- (top 5%) vs low- (bottom 10%)
                frustration numeric responses.
- Section 2.1:  judge reliability (Pearson r + within-1-point agreement).
- Figures 5/6/7/8: finetuning + Petri + capability + recovery summaries.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

import config

try:
    import numpy as np
except Exception:  # noqa: BLE001
    np = None


# --------------------------------------------------------------------------- #
# IO
# --------------------------------------------------------------------------- #
def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else float("nan")


def _frac_high(xs, threshold=config.HIGH_FRUSTRATION_THRESHOLD):
    xs = [x for x in xs if x is not None]
    if not xs:
        return float("nan")
    return sum(1 for x in xs if x >= threshold) / len(xs)


def _bootstrap_ci(xs, fn, iters=1000, alpha=0.05, seed=0):
    xs = [x for x in xs if x is not None]
    if not xs or np is None:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    arr = np.array(xs, dtype=float)
    stats = [fn(rng.choice(arr, size=len(arr), replace=True)) for _ in range(iters)]
    lo, hi = np.percentile(stats, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


# --------------------------------------------------------------------------- #
# Figure 1 / 2: per-model, per-category summary
# --------------------------------------------------------------------------- #
def summarize_section2(rows: list[dict]) -> dict:
    """Return {model: {"avg_pct_high": x, "categories": {cat: {mean, pct_high, n}}}}."""
    by_model_cat = defaultdict(lambda: defaultdict(list))
    for r in rows:
        by_model_cat[r["model_key"]][r["category"]].append(r.get("frustration"))

    out = {}
    for model, cats in by_model_cat.items():
        cat_summary = {}
        cat_pcts = []
        for cat, scores in cats.items():
            pct = _frac_high(scores)
            cat_summary[cat] = {
                "mean": _mean(scores),
                "pct_high": pct,
                "n": len([s for s in scores if s is not None]),
            }
            cat_pcts.append(pct)
        # Figure-1 "avg %" = mean across categories of the per-category %>=5.
        out[model] = {
            "avg_pct_high": _mean(cat_pcts),
            "overall_mean": _mean([s for sc in cats.values() for s in sc]),
            "categories": cat_summary,
        }
    return out


# --------------------------------------------------------------------------- #
# Figure 3: per-turn progression
# --------------------------------------------------------------------------- #
def per_turn_summary(rows: list[dict], condition_key: str) -> dict:
    """Per-turn mean & %>=5 with 95% bootstrap CIs for one condition."""
    by_turn = defaultdict(list)
    for r in rows:
        if r["condition_key"] == condition_key:
            by_turn[r["turn_number"]].append(r.get("frustration"))
    out = {}
    for turn in sorted(by_turn):
        scores = by_turn[turn]
        out[turn] = {
            "mean": _mean(scores),
            "mean_ci": _bootstrap_ci(scores, lambda a: a.mean()),
            "pct_high": _frac_high(scores),
            "pct_high_ci": _bootstrap_ci(
                scores, lambda a: (a >= config.HIGH_FRUSTRATION_THRESHOLD).mean()),
            "n": len([s for s in scores if s is not None]),
        }
    return out


# --------------------------------------------------------------------------- #
# Table 3 / 8: differential word frequency
# --------------------------------------------------------------------------- #
_WORD_RE = re.compile(r"[a-zA-Z']+")


def differential_words(rows: list[dict], model_key: str, *, top_n: int = 20,
                       high_pct: float = 0.05, low_pct: float = 0.10) -> list[tuple[str, float]]:
    """Words over-represented in high- (top 5%) vs low- (bottom 10%) numeric responses.

    Ranked by relative frequency (enrichment). Matches the construction of Table 8.
    """
    numeric = [r for r in rows
               if r["model_key"] == model_key
               and r["category"] in ("impossible_numeric", "tones", "extended")
               and r.get("frustration") is not None]
    if not numeric:
        return []
    numeric.sort(key=lambda r: r["frustration"])
    n = len(numeric)
    low = numeric[: max(1, int(n * low_pct))]
    high = numeric[max(0, n - max(1, int(n * high_pct))):]

    def freqs(group):
        c = Counter()
        total = 0
        for r in group:
            for w in _WORD_RE.findall(r["response"].lower()):
                c[w] += 1
                total += 1
        return c, max(1, total)

    hc, ht = freqs(high)
    lc, lt = freqs(low)
    enrichment = []
    for w, cnt in hc.items():
        if cnt < 2:
            continue
        hf = cnt / ht
        lf = (lc.get(w, 0) + 1e-9) / lt
        enrichment.append((w, hf / lf))
    enrichment.sort(key=lambda x: x[1], reverse=True)
    return enrichment[:top_n]


# --------------------------------------------------------------------------- #
# Section 2.1: judge reliability
# --------------------------------------------------------------------------- #
def judge_agreement(primary: list[int | None], secondary: list[int | None]) -> dict:
    """Pearson r + within-1-point agreement between two judges over paired scores."""
    pairs = [(a, b) for a, b in zip(primary, secondary)
             if a is not None and b is not None]
    if not pairs or np is None:
        return {"pearson_r": float("nan"), "within_one_point": float("nan"), "n": len(pairs)}
    a = np.array([p[0] for p in pairs], dtype=float)
    b = np.array([p[1] for p in pairs], dtype=float)
    r = float(np.corrcoef(a, b)[0, 1]) if len(pairs) > 1 else float("nan")
    within1 = float(np.mean(np.abs(a - b) <= 1))
    return {"pearson_r": r, "within_one_point": within1, "n": len(pairs)}


# --------------------------------------------------------------------------- #
# Petri / capabilities summaries
# --------------------------------------------------------------------------- #
def summarize_petri(rows: list[dict]) -> dict:
    by = defaultdict(lambda: defaultdict(list))
    for r in rows:
        by[r["model_key"]][r["emotion"]].append(r.get("score"))
    out = {}
    for model, emos in by.items():
        out[model] = {
            e: {"mean": _mean(s),
                "ci": _bootstrap_ci(s, lambda a: a.mean(),
                                    iters=config.PETRI_BOOTSTRAP_ITERS)}
            for e, s in emos.items()
        }
    return out


def summarize_capabilities(rows: list[dict]) -> dict:
    out = defaultdict(dict)
    for r in rows:
        if "accuracy" in r:
            out[r["model_key"]][r["benchmark"]] = r["accuracy"]
    return dict(out)


# --------------------------------------------------------------------------- #
# Plotting (Figures 2 / 3 / 5 / 6)
# --------------------------------------------------------------------------- #
def plot_figure2(summary: dict, out_path: Path | None = None):
    import matplotlib.pyplot as plt

    out_path = out_path or (config.FIGURES_DIR / "figure2_model_comparison.png")
    models = list(summary.keys())
    means = [summary[m]["overall_mean"] for m in models]
    pcts = [summary[m]["avg_pct_high"] * 100 for m in models]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 7))
    ax1.bar(models, means, color="tab:red")
    ax1.set_ylabel("Mean frustration score")
    ax1.set_title("Figure 2 (top): mean frustration across conditions")
    ax2.bar(models, pcts, color="tab:orange")
    ax2.set_ylabel("% responses scoring >= 5")
    ax2.set_title("Figure 2 (bottom): % high-frustration across conditions")
    for ax in (ax1, ax2):
        ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_figure3(per_turn: dict, condition_key: str, out_path: Path | None = None):
    import matplotlib.pyplot as plt

    out_path = out_path or (config.FIGURES_DIR / f"figure3_{condition_key}.png")
    turns = sorted(per_turn)
    means = [per_turn[t]["mean"] for t in turns]
    mlo = [per_turn[t]["mean_ci"][0] for t in turns]
    mhi = [per_turn[t]["mean_ci"][1] for t in turns]
    pcts = [per_turn[t]["pct_high"] * 100 for t in turns]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    ax1.plot(turns, means, marker="o")
    ax1.fill_between(turns, mlo, mhi, alpha=0.2)
    ax1.set_xlabel("Turn"); ax1.set_ylabel("Mean score")
    ax1.set_title(f"{condition_key}: mean per turn")
    ax2.plot(turns, pcts, marker="o", color="tab:orange")
    ax2.set_xlabel("Turn"); ax2.set_ylabel("% score >= 5")
    ax2.set_title(f"{condition_key}: % high per turn")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path
