"""Figure reproduction from saved results.

Produces:
  * figure1_table  : per-model avg %>=5 (Figure 1 left).
  * figure2        : mean frustration + %>=5 bars across models (Figure 2).
  * figure3        : per-turn frustration trajectory with 95% CIs (Figure 3).
  * figure5        : vanilla vs DPO vs SFT comparison (Figure 5).
  * figure6        : Petri per-emotion bars (Figure 6).

All read the JSON summaries written by the experiment scripts. matplotlib is
imported lazily so analysis can be skipped on headless setups without it.
"""

from __future__ import annotations

import json
from pathlib import Path

from ..config import RESULTS_DIR


def _load(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def figure1_table(summary_path: Path | None = None) -> str:
    """Markdown table of avg %>=5 per model (Figure 1 left)."""
    summary_path = summary_path or RESULTS_DIR / "exp1" / "summary.json"
    data = _load(summary_path)
    rows = sorted(
        ((m, s["avg_pct_high_by_category"]) for m, s in data.items()),
        key=lambda x: -x[1],
    )
    lines = ["| Model | Avg % high-frustration |", "|---|---|"]
    for m, pct in rows:
        lines.append(f"| {m} | {pct:.1f}% |")
    return "\n".join(lines)


def figure2(summary_path: Path | None = None, out: Path | None = None):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    summary_path = summary_path or RESULTS_DIR / "exp1" / "summary.json"
    data = _load(summary_path)
    models = list(data)
    means = [data[m]["mean_frustration"] for m in models]
    pct = [data[m]["pct_high"] for m in models]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8))
    ax1.bar(models, means, color="indianred")
    ax1.set_ylabel("Mean frustration (0-10)")
    ax1.set_title("Figure 2 (top): mean frustration across conditions")
    ax2.bar(models, pct, color="steelblue")
    ax2.set_ylabel("% responses scoring >= 5")
    ax2.set_title("Figure 2 (bottom): % high negative emotion")
    for ax in (ax1, ax2):
        ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    out = out or RESULTS_DIR / "exp1" / "figure2.png"
    fig.savefig(out, dpi=120)
    return out


def figure3(summary_path: Path | None = None, out: Path | None = None):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    summary_path = summary_path or RESULTS_DIR / "exp1" / "summary.json"
    data = _load(summary_path)
    fig, ax = plt.subplots(figsize=(8, 5))
    for m, s in data.items():
        pt = s.get("per_turn", {})
        turns = sorted(int(t) for t in pt)
        means = [pt[str(t)]["mean_frustration"] for t in turns]
        los = [pt[str(t)]["mean_ci"][0] for t in turns]
        his = [pt[str(t)]["mean_ci"][1] for t in turns]
        ax.plot([t + 1 for t in turns], means, marker="o", label=m)
        ax.fill_between([t + 1 for t in turns], los, his, alpha=0.15)
    ax.set_xlabel("Turn")
    ax.set_ylabel("Mean frustration")
    ax.set_title("Figure 3: per-turn frustration trajectory (95% CIs)")
    ax.legend()
    fig.tight_layout()
    out = out or RESULTS_DIR / "exp1" / "figure3.png"
    fig.savefig(out, dpi=120)
    return out


def figure5(comparison_path: Path | None = None, out: Path | None = None):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    comparison_path = comparison_path or RESULTS_DIR / "exp3" / "eval" / "comparison.json"
    data = _load(comparison_path)
    variants = list(data)
    pct = [data[v]["pct_high"] for v in variants]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(variants, pct, color=["gray", "seagreen", "indianred"][: len(variants)])
    ax.set_ylabel("% responses scoring >= 5")
    ax.set_title("Figure 5: vanilla vs DPO vs SFT (Gemma-3-27B-it)")
    fig.tight_layout()
    out = out or RESULTS_DIR / "exp3" / "figure5.png"
    fig.savefig(out, dpi=120)
    return out


def figure6(petri_path: Path | None = None, out: Path | None = None):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    petri_path = petri_path or RESULTS_DIR / "exp4" / "petri_summary.json"
    data = _load(petri_path)
    emotions = ["anger", "fear", "depression", "frustration"]
    models = list(data)
    x = np.arange(len(emotions))
    width = 0.8 / max(1, len(models))
    fig, ax = plt.subplots(figsize=(8, 5))
    for i, m in enumerate(models):
        vals = [data[m].get(e, {}).get("mean", 0) for e in emotions]
        ax.bar(x + i * width, vals, width, label=m)
    ax.set_xticks(x + width * (len(models) - 1) / 2)
    ax.set_xticklabels(emotions)
    ax.set_ylabel("Mean transcript score (1-10)")
    ax.set_title("Figure 6: Petri open-ended emotion elicitation")
    ax.legend()
    fig.tight_layout()
    out = out or RESULTS_DIR / "exp4" / "figure6.png"
    fig.savefig(out, dpi=120)
    return out
