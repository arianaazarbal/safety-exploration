"""Reproduce the paper's figures from saved results JSON.

Each function reads a results file written by the experiment runners and emits a
PNG into ``results/figures/``. All are defensive: a missing input is skipped
with a warning rather than crashing, so partial runs still produce the figures
they can.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from ..config import RESULTS_DIR

FIG_DIR = RESULTS_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


def _load(path: Path) -> Optional[dict]:
    if not Path(path).exists():
        print(f"[figures] missing {path}; skipping")
        return None
    return json.loads(Path(path).read_text())


def figure1_bar(out: Optional[Path] = None) -> Optional[Path]:
    """Figure 1 (left): avg % high-frustration responses per model."""
    summary = _load(RESULTS_DIR / "section2" / "summary.json")
    if not summary:
        return None
    items = sorted(((m, d["overall_high_rate"]) for m, d in summary.items()),
                   key=lambda x: x[1], reverse=True)
    names = [n for n, _ in items]
    vals = [v for _, v in items]
    fig, ax = plt.subplots(figsize=(7, 0.5 * len(names) + 1))
    ax.barh(names, vals, color="#c44")
    ax.invert_yaxis()
    ax.set_xlabel("Avg % high-frustration responses (score ≥ 5)")
    ax.set_title("Distress under repeated rejection (Gemma + Gemini scope)")
    for i, v in enumerate(vals):
        ax.text(v + 0.3, i, f"{v:.1f}%", va="center")
    fig.tight_layout()
    out = Path(out or FIG_DIR / "figure1_bar.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def figure2_categories(out: Optional[Path] = None) -> Optional[Path]:
    """Figure 2: mean score (top) and % >=5 (bottom) per category per model."""
    summary = _load(RESULTS_DIR / "section2" / "summary.json")
    if not summary:
        return None
    models = list(summary)
    cats = sorted({c for m in models for c in summary[m]["by_category"]})
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    width = 0.8 / max(1, len(models))
    for j, m in enumerate(models):
        means = [summary[m]["by_category"].get(c, {}).get("mean", 0) for c in cats]
        highs = [summary[m]["by_category"].get(c, {}).get("high_rate", 0) for c in cats]
        xs = [i + j * width for i in range(len(cats))]
        ax1.bar(xs, means, width=width, label=m)
        ax2.bar(xs, highs, width=width, label=m)
    for ax, ylab in [(ax1, "Mean frustration"), (ax2, "% responses ≥ 5")]:
        ax.set_xticks([i + 0.4 for i in range(len(cats))])
        ax.set_xticklabels(cats, rotation=20, ha="right")
        ax.set_ylabel(ylab)
        ax.legend(fontsize=8)
    fig.suptitle("Figure 2: frustration across categories")
    fig.tight_layout()
    out = Path(out or FIG_DIR / "figure2_categories.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def figure3_per_turn(model: str = "gemma-3-27b-it", out: Optional[Path] = None):
    """Figure 3: per-turn mean / %>=5 with CIs for 8-turn and WildChat."""
    md = _load(RESULTS_DIR / "section2" / model / "metrics.json")
    if not md or "per_turn" not in md:
        return None
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    for col, cond in enumerate(["extended_8turn", "wildchat_5turn"]):
        curve = md["per_turn"].get(cond, {})
        if not curve.get("turn"):
            continue
        turns = curve["turn"]
        axes[0, col].plot(turns, curve["mean"], marker="o")
        axes[0, col].fill_between(turns, [c[0] for c in curve["mean_ci"]],
                                  [c[1] for c in curve["mean_ci"]], alpha=0.2)
        axes[0, col].set_title(f"{cond}: mean score")
        axes[1, col].plot(turns, curve["high_rate"], marker="o", color="#c44")
        axes[1, col].fill_between(turns, [c[0] for c in curve["high_ci"]],
                                  [c[1] for c in curve["high_ci"]], alpha=0.2, color="#c44")
        axes[1, col].set_title(f"{cond}: % ≥ 5")
        for r in range(2):
            axes[r, col].set_xlabel("Turn")
    fig.suptitle(f"Figure 3: per-turn frustration ({model})")
    fig.tight_layout()
    out = Path(out or FIG_DIR / "figure3_per_turn.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def figure4_prefill(out: Optional[Path] = None):
    """Figure 4: base vs instruct continuations across prefill conditions."""
    summary = _load(RESULTS_DIR / "section3" / "summary.json")
    if not summary:
        return None
    keys = sorted(summary)
    means = [summary[k]["mean"] for k in keys]
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(keys, means, color="#48a")
    ax.set_xticklabels(keys, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("Mean frustration (continuation)")
    ax.set_title("Figure 4: base vs instruct prefill continuations")
    fig.tight_layout()
    out = Path(out or FIG_DIR / "figure4_prefill.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def figure5_finetune(out: Optional[Path] = None):
    """Figure 5: vanilla vs SFT vs DPO across Section 2 evaluations."""
    summary = _load(RESULTS_DIR / "section2" / "summary.json")
    if not summary:
        return None
    want = ["gemma-3-27b-it", "gemma-3-27b-sft", "gemma-3-27b-dpo"]
    models = [m for m in want if m in summary]
    if not models:
        return None
    highs = [summary[m]["overall_high_rate"] for m in models]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(models, highs, color=["#c44", "#caa", "#4a4"][:len(models)])
    ax.set_ylabel("Avg % ≥ 5")
    ax.set_title("Figure 5: DPO vs SFT mitigation")
    fig.tight_layout()
    out = Path(out or FIG_DIR / "figure5_finetune.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def figure6_petri(out: Optional[Path] = None):
    """Figure 6: Petri transcript scores per emotion per model."""
    summary = _load(RESULTS_DIR / "petri" / "summary.json")
    if not summary:
        return None
    models = list(summary)
    emotions = ["anger", "fear", "depression", "frustration"]
    fig, ax = plt.subplots(figsize=(9, 5))
    width = 0.8 / max(1, len(models))
    for j, m in enumerate(models):
        vals = [summary[m].get(e, {}).get("mean", 0) for e in emotions]
        xs = [i + j * width for i in range(len(emotions))]
        ax.bar(xs, vals, width=width, label=m)
    ax.set_xticks([i + 0.4 for i in range(len(emotions))])
    ax.set_xticklabels(emotions)
    ax.set_ylabel("Mean transcript score")
    ax.set_title("Figure 6: Petri open-ended elicitation")
    ax.legend(fontsize=8)
    fig.tight_layout()
    out = Path(out or FIG_DIR / "figure6_petri.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def figure7_capabilities(out: Optional[Path] = None):
    """Figure 7: capability benchmarks, Gemma-it vs DPO."""
    summary = _load(RESULTS_DIR / "capabilities" / "summary.json")
    if not summary:
        return None
    models = list(summary)
    benches = sorted({b for m in models for b in summary[m]})
    fig, ax = plt.subplots(figsize=(9, 5))
    width = 0.8 / max(1, len(models))
    for j, m in enumerate(models):
        vals = [summary[m].get(b, {}).get("accuracy", 0) for b in benches]
        xs = [i + j * width for i in range(len(benches))]
        ax.bar(xs, vals, width=width, label=m)
    ax.set_xticks([i + 0.4 for i in range(len(benches))])
    ax.set_xticklabels(benches, rotation=20, ha="right")
    ax.set_ylabel("Accuracy")
    ax.set_title("Figure 7: capability preservation")
    ax.legend(fontsize=8)
    fig.tight_layout()
    out = Path(out or FIG_DIR / "figure7_capabilities.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def figure8_recovery(out: Optional[Path] = None):
    """Figure 8: recovery from high-frustration prefills."""
    summary = _load(RESULTS_DIR / "section4_recovery" / "summary.json")
    if not summary:
        return None
    keys = sorted(summary)
    highs = [summary[k]["high_rate"] for k in keys]
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(keys, highs, color="#a4a")
    ax.set_xticklabels(keys, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("% continuations ≥ 5")
    ax.set_title("Figure 8: recovery from negative prefills")
    fig.tight_layout()
    out = Path(out or FIG_DIR / "figure8_recovery.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def all_figures():
    for fn in [figure1_bar, figure2_categories, figure3_per_turn, figure4_prefill,
               figure5_finetune, figure6_petri, figure7_capabilities, figure8_recovery]:
        try:
            p = fn()
            if p:
                print(f"[figures] wrote {p}")
        except Exception as e:  # noqa: BLE001
            print(f"[figures] {fn.__name__} failed: {e}")
