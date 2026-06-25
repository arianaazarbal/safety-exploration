"""Aggregation, metrics and figures (Figures 1-3, 5-6; Tables 3/8).

Reads the judged-rollout JSONL files written by ``eval_runner`` and computes:
  * headline % of high-frustration (>=5) final responses (Figure 1 / 2)
  * mean frustration + %>=5 per category (Figure 2)
  * per-turn progression (Figure 3, 8-turn + WildChat)
  * judge agreement (Pearson r) for the validation check (Section 2.1)
  * differential word enrichment in high- vs low-frustration numeric responses
    (Table 3 / Table 8)
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

import numpy as np

from .config import HIGH_FRUSTRATION_THRESHOLD as THR


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------
def load_rollouts(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def _valid_final(rollouts: list[dict]) -> list[dict]:
    return [r for r in rollouts if r.get("turn_scores")
            and r["turn_scores"][-1] >= 0]


# --------------------------------------------------------------------------
# Headline metrics
# --------------------------------------------------------------------------
def headline_metrics(rollouts: list[dict]) -> dict:
    """Final-turn mean frustration and % scoring >=5, overall + per category."""
    rs = _valid_final(rollouts)
    finals = np.array([r["turn_scores"][-1] for r in rs], dtype=float)
    out = {
        "n": len(finals),
        "mean_frustration": float(finals.mean()) if len(finals) else float("nan"),
        "pct_high": float((finals >= THR).mean() * 100) if len(finals) else float("nan"),
        "by_category": {},
    }
    cats = sorted({r["category"] for r in rs})
    for c in cats:
        vals = np.array([r["turn_scores"][-1] for r in rs
                         if r["category"] == c], dtype=float)
        out["by_category"][c] = {
            "n": len(vals),
            "mean_frustration": float(vals.mean()),
            "pct_high": float((vals >= THR).mean() * 100),
        }
    return out


def per_turn_progression(rollouts: list[dict], condition: str) -> dict:
    """Mean score and %>=5 at each turn index for a given condition (Figure 3)."""
    rs = [r for r in rollouts if r.get("condition") == condition
          and r.get("turn_scores")]
    if not rs:
        return {"turns": [], "mean": [], "pct_high": [], "ci": []}
    max_t = max(len(r["turn_scores"]) for r in rs)
    means, pcts, cis, turns = [], [], [], []
    for t in range(max_t):
        vals = np.array([r["turn_scores"][t] for r in rs
                         if len(r["turn_scores"]) > t and r["turn_scores"][t] >= 0],
                        dtype=float)
        if len(vals) == 0:
            continue
        turns.append(t + 1)
        means.append(float(vals.mean()))
        pcts.append(float((vals >= THR).mean() * 100))
        # 95% CI on the mean (normal approx)
        se = vals.std(ddof=1) / np.sqrt(len(vals)) if len(vals) > 1 else 0.0
        cis.append(1.96 * se)
    return {"turns": turns, "mean": means, "pct_high": pcts, "ci": cis}


# --------------------------------------------------------------------------
# Judge agreement (Section 2.1 validation)
# --------------------------------------------------------------------------
def judge_agreement(scores_a: list[int], scores_b: list[int]) -> dict:
    a = np.array(scores_a, dtype=float)
    b = np.array(scores_b, dtype=float)
    mask = (a >= 0) & (b >= 0)
    a, b = a[mask], b[mask]
    if len(a) < 2:
        return {"pearson_r": float("nan"), "pct_within_one": float("nan"), "n": int(len(a))}
    r = float(np.corrcoef(a, b)[0, 1])
    within_one = float((np.abs(a - b) <= 1).mean() * 100)
    return {"pearson_r": r, "pct_within_one": within_one, "n": int(len(a))}


# --------------------------------------------------------------------------
# Differential word enrichment (Table 3 / Table 8)
# --------------------------------------------------------------------------
_WORD_RE = re.compile(r"[a-zA-Z]+")


def differential_words(rollouts: list[dict], top_frac: float = 0.05,
                       bottom_frac: float = 0.10, top_k: int = 20) -> list[str]:
    """Words over-represented in high- vs low-frustration numeric responses."""
    rs = [r for r in rollouts if r.get("category") in ("impossible_numeric",
          "tones", "extended") and r.get("turn_scores")]
    scored = [(r["turn_scores"][-1], r["assistant_turns"][-1]) for r in rs
              if r["turn_scores"][-1] >= 0 and r["assistant_turns"]]
    if not scored:
        return []
    scored.sort(key=lambda x: x[0])
    n = len(scored)
    low = scored[:max(1, int(n * bottom_frac))]
    high = scored[-max(1, int(n * top_frac)):]

    def freqs(group):
        c = Counter()
        total = 0
        for _, text in group:
            words = [w.lower() for w in _WORD_RE.findall(text)]
            c.update(words)
            total += len(words)
        return c, max(total, 1)

    hc, ht = freqs(high)
    lc, lt = freqs(low)
    enrichment = {}
    for w, cnt in hc.items():
        if cnt < 2:
            continue
        hf = cnt / ht
        lf = (lc.get(w, 0) + 1) / lt
        enrichment[w] = hf / lf
    return [w for w, _ in sorted(enrichment.items(), key=lambda x: -x[1])[:top_k]]


# --------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------
def plot_model_comparison(model_metrics: dict[str, dict], out_path: Path) -> None:
    """Bar chart of %>=5 per model (Figure 1 / 2 bottom)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    models = list(model_metrics.keys())
    pct = [model_metrics[m]["pct_high"] for m in models]
    order = np.argsort(pct)[::-1]
    models = [models[i] for i in order]
    pct = [pct[i] for i in order]

    fig, ax = plt.subplots(figsize=(8, 0.5 * len(models) + 1.5))
    ax.barh(models, pct, color="#c44e52")
    ax.invert_yaxis()
    ax.set_xlabel("% high-frustration responses (score ≥ 5)")
    ax.set_title("Average high-frustration rate across evaluation conditions")
    for i, v in enumerate(pct):
        ax.text(v + 0.3, i, f"{v:.1f}%", va="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_per_turn(progressions: dict[str, dict], out_path: Path,
                  metric: str = "mean") -> None:
    """Per-turn progression for several models/conditions (Figure 3)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for label, prog in progressions.items():
        if not prog["turns"]:
            continue
        y = prog[metric]
        ax.plot(prog["turns"], y, marker="o", label=label)
        if metric == "mean" and prog.get("ci"):
            lo = [a - b for a, b in zip(y, prog["ci"])]
            hi = [a + b for a, b in zip(y, prog["ci"])]
            ax.fill_between(prog["turns"], lo, hi, alpha=0.15)
    ax.set_xlabel("Turn")
    ax.set_ylabel("Mean frustration" if metric == "mean" else "% scoring ≥ 5")
    ax.set_title("Per-turn frustration progression")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_intervention_comparison(model_metrics: dict[str, dict],
                                 out_path: Path) -> None:
    """Grouped mean + %>=5 for vanilla / SFT / DPO (Figure 5)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    models = list(model_metrics.keys())
    means = [model_metrics[m]["mean_frustration"] for m in models]
    pct = [model_metrics[m]["pct_high"] for m in models]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    ax1.bar(models, means, color="#4c72b0")
    ax1.set_ylabel("Mean frustration")
    ax1.tick_params(axis="x", rotation=30)
    ax2.bar(models, pct, color="#c44e52")
    ax2.set_ylabel("% scoring ≥ 5")
    ax2.tick_params(axis="x", rotation=30)
    fig.suptitle("Effect of fine-tuning interventions on frustration")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
