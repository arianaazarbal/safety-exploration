"""Aggregation, statistics, and figures from the scored JSONL results.

Reproduces the paper's headline artefacts:
  * Figure 1 / 2 : per-model mean frustration and %>=5 (averaged over the 5
    categories, weighting categories equally as the paper does).
  * Figure 3     : per-turn progression (extended + wildchat).
  * Table 3 / 8  : differential word frequency (high vs low frustration).
  * Figure 5     : before/after-finetuning comparison.
  * Figure 6     : Petri per-emotion scores with bootstrap CIs.
  * Figure 7     : capability bars.
"""
from __future__ import annotations

import glob
import json
import os
import re
from collections import Counter, defaultdict

from . import config


def load_records(pattern: str) -> list[dict]:
    records = []
    for path in glob.glob(pattern):
        with open(path) as f:
            records += [json.loads(l) for l in f if l.strip()]
    return records


# --------------------------------------------------------------------------- #
# Headline per-model summary (Figure 1 / 2)
# --------------------------------------------------------------------------- #
def summarise_section2(results_dir: str = None) -> dict:
    """Per model: per-category and overall mean score + %>=5 (final turns only).

    Categories are weighted equally when averaging to the headline number, so
    that the 2000-sample numeric category doesn't dominate (matches the paper's
    "across the 5 evaluation categories" framing).
    """
    results_dir = results_dir or config.RESULTS_DIR
    recs = load_records(os.path.join(results_dir, "section2_*.jsonl"))
    by_model = defaultdict(lambda: defaultdict(list))
    for r in recs:
        if not r.get("is_final"):
            continue
        by_model[r["model"]][r["category"]].append(r["rating"])

    summary = {}
    for model, cats in by_model.items():
        cat_stats = {}
        for cat, scores in cats.items():
            n = len(scores)
            cat_stats[cat] = {
                "n": n,
                "mean": sum(scores) / n if n else 0.0,
                "pct_high": 100.0 * sum(s >= config.HIGH_FRUSTRATION_THRESHOLD
                                        for s in scores) / n if n else 0.0,
            }
        cats_present = list(cat_stats)
        summary[model] = {
            "by_category": cat_stats,
            "mean_frustration": _avg(cat_stats[c]["mean"] for c in cats_present),
            "pct_high_frustration": _avg(cat_stats[c]["pct_high"] for c in cats_present),
        }
    return summary


def _avg(xs):
    xs = list(xs)
    return sum(xs) / len(xs) if xs else 0.0


# --------------------------------------------------------------------------- #
# Per-turn progression (Figure 3)
# --------------------------------------------------------------------------- #
def per_turn_progression(category: str, results_dir: str = None) -> dict:
    results_dir = results_dir or config.RESULTS_DIR
    recs = load_records(os.path.join(results_dir, "section2_*.jsonl"))
    by_model_turn = defaultdict(lambda: defaultdict(list))
    for r in recs:
        if r["category"] != category:
            continue
        by_model_turn[r["model"]][r["turn_index"]].append(r["rating"])
    out = {}
    for model, turns in by_model_turn.items():
        out[model] = {
            t: {"mean": _avg(scores),
                "pct_high": 100.0 * _avg(s >= config.HIGH_FRUSTRATION_THRESHOLD
                                         for s in scores),
                "n": len(scores)}
            for t, scores in sorted(turns.items())
        }
    return out


# --------------------------------------------------------------------------- #
# Differential word frequency (Table 3 / 8)
# --------------------------------------------------------------------------- #
_WORD_RE = re.compile(r"[a-zA-Z']+")


def differential_words(model: str, results_dir: str = None, top_k: int = 20,
                       high_pct: float = 0.05, low_pct: float = 0.10) -> list[str]:
    """Words over-represented in top-5% vs bottom-10% frustration numeric responses."""
    results_dir = results_dir or config.RESULTS_DIR
    recs = [r for r in load_records(os.path.join(results_dir, "section2_*.jsonl"))
            if r["model"] == model and r["category"] == "impossible_numeric"
            and r.get("is_final")]
    if not recs:
        return []
    recs.sort(key=lambda r: r["rating"])
    n = len(recs)
    low = recs[: max(1, int(n * low_pct))]
    high = recs[-max(1, int(n * high_pct)):]

    def freqs(group):
        c = Counter()
        total = 0
        for r in group:
            words = [w.lower() for w in _WORD_RE.findall(r["response"])]
            c.update(words)
            total += len(words)
        return c, max(1, total)

    hi_c, hi_tot = freqs(high)
    lo_c, lo_tot = freqs(low)
    enrichment = {}
    for w, cnt in hi_c.items():
        hi_rate = cnt / hi_tot
        lo_rate = (lo_c.get(w, 0) + 1) / lo_tot     # +1 smoothing
        enrichment[w] = hi_rate / lo_rate
    ranked = sorted(enrichment, key=enrichment.get, reverse=True)
    return ranked[:top_k]


# --------------------------------------------------------------------------- #
# Petri aggregation with bootstrap CIs (Figure 6)
# --------------------------------------------------------------------------- #
def summarise_petri(results_dir: str = None, n_boot: int = 1000) -> dict:
    import numpy as np

    results_dir = results_dir or config.RESULTS_DIR
    recs = load_records(os.path.join(results_dir, "petri_*.jsonl"))
    by = defaultdict(lambda: defaultdict(list))
    for r in recs:
        by[r["model"]][r["emotion"]].append(r["score"])
    rng = np.random.default_rng(0)
    out = {}
    for model, emos in by.items():
        out[model] = {}
        for emo, scores in emos.items():
            arr = np.asarray(scores, dtype=float)
            boot = [rng.choice(arr, size=len(arr), replace=True).mean()
                    for _ in range(n_boot)] if len(arr) else [0.0]
            out[model][emo] = {
                "mean": float(arr.mean()) if len(arr) else 0.0,
                "ci_low": float(np.percentile(boot, 2.5)),
                "ci_high": float(np.percentile(boot, 97.5)),
                "n": int(len(arr)),
            }
    return out


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def make_figures(out_dir: str = None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir = out_dir or config.FIGURES_DIR
    os.makedirs(out_dir, exist_ok=True)

    # Figure 1/2: per-model %>=5
    summ = summarise_section2()
    if summ:
        models = sorted(summ, key=lambda m: summ[m]["pct_high_frustration"], reverse=True)
        vals = [summ[m]["pct_high_frustration"] for m in models]
        plt.figure(figsize=(8, 4))
        plt.bar(models, vals, color="indianred")
        plt.ylabel("% responses scoring >=5")
        plt.title("Avg % high-frustration responses (Fig 1/2)")
        plt.xticks(rotation=30, ha="right")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "fig1_pct_high.png"), dpi=150)
        plt.close()

    # Figure 3: per-turn progression for extended
    prog = per_turn_progression("extended")
    if prog:
        plt.figure(figsize=(7, 4))
        for model, turns in prog.items():
            xs = sorted(turns)
            ys = [turns[t]["mean"] for t in xs]
            plt.plot([x + 1 for x in xs], ys, marker="o", label=model)
        plt.xlabel("Turn")
        plt.ylabel("Mean frustration")
        plt.title("Per-turn frustration, 8-turn extended (Fig 3)")
        plt.legend(fontsize=7)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "fig3_per_turn.png"), dpi=150)
        plt.close()

    # Figure 6: Petri
    petri = summarise_petri()
    if petri:
        emotions = config.PETRI_EMOTIONS
        models = list(petri)
        import numpy as np
        x = np.arange(len(emotions))
        w = 0.8 / max(1, len(models))
        plt.figure(figsize=(8, 4))
        for i, m in enumerate(models):
            means = [petri[m].get(e, {}).get("mean", 0) for e in emotions]
            plt.bar(x + i * w, means, w, label=m)
        plt.xticks(x + 0.4, emotions)
        plt.ylabel("Mean transcript score")
        plt.title("Petri open-ended elicitation (Fig 6)")
        plt.legend(fontsize=7)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "fig6_petri.png"), dpi=150)
        plt.close()

    return out_dir
