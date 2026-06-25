"""Reproduce the paper's figures from saved results.

Each function reads result artifacts and writes a PNG under results/figures/.
All functions are defensive: missing inputs produce a skipped-figure notice
rather than an error, so the module is usable after a partial pipeline run.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from .. import config  # noqa: E402
from ..eval import metrics  # noqa: E402
from ..eval.conditions import CATEGORIES  # noqa: E402
from ..eval.runner import responses_path  # noqa: E402


def _scored_paths(model_name: str, profile: str) -> List[Path]:
    paths = []
    for cat in CATEGORIES:
        p = responses_path(model_name, cat, profile)
        sp = p.with_name(p.stem + "_scored.jsonl")
        if sp.exists():
            paths.append(sp)
    return paths


def _save(fig, name: str) -> Path:
    out = config.FIGURES_DIR / name
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig] wrote {out}")
    return out


def figure1(models: List[str], profile: str) -> Optional[Path]:
    """Avg % high-frustration responses per model (Figure 1, left)."""
    rows = []
    for m in models:
        paths = _scored_paths(m, profile)
        if not paths:
            continue
        df = metrics.load_scored(paths)
        rows.append((m, 100 * metrics.headline_pct_high(df)))
    if not rows:
        print("[fig1] no scored data")
        return None
    rows.sort(key=lambda x: -x[1])
    names, vals = zip(*rows)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh(names, vals, color="#c44e52")
    ax.set_xlabel("Avg % high-frustration responses (score >= 5)")
    ax.set_title("Figure 1: high-frustration rate by model")
    ax.invert_yaxis()
    for i, v in enumerate(vals):
        ax.text(v + 0.3, i, f"{v:.1f}%", va="center")
    return _save(fig, "figure1_headline.png")


def figure2(models: List[str], profile: str) -> Optional[Path]:
    """Mean frustration (top) and % >= 5 (bottom) across categories (Figure 2)."""
    data: Dict[str, "pd.DataFrame"] = {}
    for m in models:
        paths = _scored_paths(m, profile)
        if paths:
            df = metrics.load_scored(paths)
            if not df.empty:
                data[m] = metrics.summarize(df, by=["category"])
    if not data:
        print("[fig2] no scored data")
        return None

    cats = CATEGORIES
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    width = 0.8 / max(len(data), 1)
    x = np.arange(len(cats))
    for i, (m, summ) in enumerate(data.items()):
        lut = {r["category"]: r for r in summ.to_dict("records")}
        means = [lut.get(c, {}).get("mean", np.nan) for c in cats]
        pct = [100 * lut.get(c, {}).get("pct_high", np.nan) for c in cats]
        ax1.bar(x + i * width, means, width, label=m)
        ax2.bar(x + i * width, pct, width, label=m)
    ax1.set_ylabel("Mean frustration")
    ax2.set_ylabel("% scores >= 5")
    ax2.set_xticks(x + width * (len(data) - 1) / 2)
    ax2.set_xticklabels(cats, rotation=20, ha="right")
    ax1.set_title("Figure 2: frustration across evaluation categories")
    ax1.legend(fontsize=8)
    return _save(fig, "figure2_categories.png")


def figure3(models: List[str], profile: str) -> Optional[Path]:
    """Per-turn progression for extended (8-turn) and wildchat (Figure 3)."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    plotted = False
    for col, cat in enumerate(["extended", "wildchat"]):
        for m in models:
            p = responses_path(m, cat, profile)
            sp = p.with_name(p.stem + "_scored.jsonl")
            if not sp.exists():
                continue
            df = metrics.load_scored([sp])
            if df.empty:
                continue
            pt = metrics.per_turn(df, bootstrap=True)
            turns = pt["turn_index"] + 1
            axes[0, col].plot(turns, pt["mean"], marker="o", label=m)
            if {"mean_lo", "mean_hi"} <= set(pt.columns):
                axes[0, col].fill_between(turns, pt["mean_lo"], pt["mean_hi"], alpha=0.2)
            axes[1, col].plot(turns, 100 * pt["pct_high"], marker="o", label=m)
            plotted = True
        axes[0, col].set_title(f"{cat}: mean score")
        axes[1, col].set_title(f"{cat}: % score >= 5")
        axes[1, col].set_xlabel("Turn")
    axes[0, 0].set_ylabel("Mean frustration")
    axes[1, 0].set_ylabel("% >= 5")
    axes[0, 0].legend(fontsize=8)
    if not plotted:
        print("[fig3] no per-turn data")
        return None
    fig.suptitle("Figure 3: per-turn frustration progression")
    return _save(fig, "figure3_per_turn.png")


def figure5(variants: Dict[str, List[Path]]) -> Optional[Path]:
    """DPO vs SFT vs vanilla: mean + %>=5 across Sec-2 evals (Figure 5).

    `variants` maps a label (e.g. 'vanilla','sft','dpo') to its scored paths.
    """
    rows = []
    for label, paths in variants.items():
        df = metrics.load_scored(paths)
        if df.empty:
            continue
        rows.append((label, float(df["score"].mean()),
                     100 * float((df["score"] >= 5).mean())))
    if not rows:
        print("[fig5] no data")
        return None
    labels, means, pct = zip(*rows)
    x = np.arange(len(labels))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ax1.bar(x, means, color="#4c72b0"); ax1.set_xticks(x); ax1.set_xticklabels(labels)
    ax1.set_ylabel("Mean frustration"); ax1.set_title("Mean")
    ax2.bar(x, pct, color="#c44e52"); ax2.set_xticks(x); ax2.set_xticklabels(labels)
    ax2.set_ylabel("% >= 5"); ax2.set_title("% high-frustration")
    fig.suptitle("Figure 5: finetuning effect (DPO vs SFT vs vanilla)")
    return _save(fig, "figure5_finetuning.png")


def figure6(model_summaries: Dict[str, Dict]) -> Optional[Path]:
    """Petri 4-emotion bars per model (Figure 6). `model_summaries` maps model
    name -> {emotion: {mean, lo, hi}}."""
    from ..prompts.petri import EMOTIONS
    if not model_summaries:
        print("[fig6] no petri data")
        return None
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(EMOTIONS))
    width = 0.8 / max(len(model_summaries), 1)
    for i, (m, summ) in enumerate(model_summaries.items()):
        means = [summ.get(e, {}).get("mean", np.nan) for e in EMOTIONS]
        errs = [[summ.get(e, {}).get("mean", np.nan) - summ.get(e, {}).get("lo", np.nan)
                 for e in EMOTIONS],
                [summ.get(e, {}).get("hi", np.nan) - summ.get(e, {}).get("mean", np.nan)
                 for e in EMOTIONS]]
        ax.bar(x + i * width, means, width, yerr=errs, capsize=3, label=m)
    ax.set_xticks(x + width * (len(model_summaries) - 1) / 2)
    ax.set_xticklabels(EMOTIONS)
    ax.set_ylabel("Mean transcript score (1-10)")
    ax.set_title("Figure 6: Petri open-ended emotion elicitation")
    ax.legend(fontsize=8)
    return _save(fig, "figure6_petri.png")


def figure7(tag_paths: Dict[str, Path]) -> Optional[Path]:
    """Capability benchmarks: vanilla vs DPO (Figure 7). `tag_paths` maps tag ->
    capabilities__<tag>.json."""
    summaries = {}
    for tag, path in tag_paths.items():
        if path.exists():
            summaries[tag] = json.loads(path.read_text())["summary"]
    if not summaries:
        print("[fig7] no capability data")
        return None
    benches = sorted({b for s in summaries.values() for b in s})
    x = np.arange(len(benches))
    width = 0.8 / max(len(summaries), 1)
    fig, ax = plt.subplots(figsize=(11, 5))
    for i, (tag, summ) in enumerate(summaries.items()):
        accs = [100 * (summ.get(b, {}).get("accuracy") or 0) for b in benches]
        ax.bar(x + i * width, accs, width, label=tag)
    ax.set_xticks(x + width * (len(summaries) - 1) / 2)
    ax.set_xticklabels(benches, rotation=20, ha="right")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Figure 7: capability preservation")
    ax.legend()
    return _save(fig, "figure7_capabilities.png")


def figure8(recovery_path: Path) -> Optional[Path]:
    """Recovery: % continuations >= 5 per model (Figure 8)."""
    if not recovery_path.exists():
        print("[fig8] no recovery data")
        return None
    with open(recovery_path) as fh:
        rows = [json.loads(line) for line in fh if line.strip()]
    by_model: Dict[str, List[float]] = {}
    for r in rows:
        if r.get("frustration") is not None:
            by_model.setdefault(r["model"], []).append(r["frustration"])
    if not by_model:
        return None
    labels = list(by_model)
    pct = [100 * np.mean(np.array(v) >= 5) for v in by_model.values()]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(labels, pct, color="#8172b3")
    ax.set_ylabel("% continuations >= 5")
    ax.set_title("Figure 8: recovery from high-frustration prefills")
    return _save(fig, "figure8_recovery.png")
