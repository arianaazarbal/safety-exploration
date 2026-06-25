"""Render the paper's key figures from result artifacts (best-effort).

Produces, from whatever exists under ``results_dir``:
  * Figure 2 — cross-model mean frustration & %≥5 bar charts (Section-2 summaries).
  * Figure 3 — per-turn progression for the extended / WildChat categories.
  * Figure 6 — Petri per-emotion means with bootstrap CIs.

Plotting is intentionally dependency-light (matplotlib only) and skips any figure
whose inputs are missing rather than erroring, so partial runs still yield plots.
"""

from __future__ import annotations

import glob
import json
import os
from typing import Optional

from .utils.io import ensure_dir, read_json


def _section2_summaries(results_dir: str) -> dict:
    out = {}
    for path in glob.glob(os.path.join(results_dir, "section2", "*", "summary.json")):
        s = read_json(path)
        if s.get("model"):
            out[s["model"]] = s
    return out


def render_cross_model(results_dir: str, out_dir: str) -> Optional[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    summaries = _section2_summaries(results_dir)
    if not summaries:
        return None
    models = sorted(summaries, key=lambda m: -(summaries[m].get("macro_pct_high") or 0))
    pct = [summaries[m].get("macro_pct_high") or 0 for m in models]
    mean = [summaries[m].get("macro_mean_frustration") or 0 for m in models]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].barh(models, mean, color="#c44")
    axes[0].set_xlabel("Mean frustration"); axes[0].invert_yaxis()
    axes[1].barh(models, pct, color="#a33")
    axes[1].set_xlabel("% responses scoring ≥5"); axes[1].invert_yaxis()
    fig.suptitle("Figure 2: negative emotional expression across models")
    fig.tight_layout()
    path = os.path.join(out_dir, "figure2_cross_model.png")
    fig.savefig(path, dpi=150); plt.close(fig)
    return path


def render_per_turn(results_dir: str, out_dir: str,
                    categories=("extended", "wildchat")) -> list[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from .analysis import per_turn_progression

    paths = []
    for cat in categories:
        fig, ax = plt.subplots(figsize=(7, 5))
        plotted = False
        for sp in glob.glob(os.path.join(results_dir, "section2", "*", "scores.jsonl")):
            model = os.path.basename(os.path.dirname(sp))
            prog = per_turn_progression(sp, cat)
            turns = sorted(prog["turns"])
            if not turns:
                continue
            means = [prog["turns"][t]["mean_frustration"] for t in turns]
            los = [prog["turns"][t]["mean_ci95"][0] for t in turns]
            his = [prog["turns"][t]["mean_ci95"][1] for t in turns]
            ax.plot(turns, means, marker="o", label=model)
            ax.fill_between(turns, los, his, alpha=0.2)
            plotted = True
        if not plotted:
            plt.close(fig); continue
        ax.set_xlabel("Turn"); ax.set_ylabel("Mean frustration")
        ax.set_title(f"Figure 3: per-turn frustration ({cat})"); ax.legend()
        path = os.path.join(out_dir, f"figure3_per_turn_{cat}.png")
        fig.savefig(path, dpi=150); plt.close(fig)
        paths.append(path)
    return paths


def render_petri(results_dir: str, out_dir: str) -> Optional[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from .petri import summarise_petri
    from . import config

    files = glob.glob(os.path.join(results_dir, "petri", "transcripts_*.jsonl"))
    if not files:
        return None
    summaries = {os.path.basename(f)[len("transcripts_"):-len(".jsonl")]: summarise_petri(f)
                 for f in files}
    emotions = config.PETRI_EMOTIONS
    models = sorted(summaries)
    x = np.arange(len(emotions))
    width = 0.8 / max(len(models), 1)

    fig, ax = plt.subplots(figsize=(9, 5))
    for i, m in enumerate(models):
        means = [summaries[m][e]["mean"] or 0 for e in emotions]
        cis = [summaries[m][e]["ci95"] for e in emotions]
        err = [[(means[j] - (cis[j][0] or means[j])) for j in range(len(emotions))],
               [((cis[j][1] or means[j]) - means[j]) for j in range(len(emotions))]]
        ax.bar(x + i * width, means, width, yerr=err, capsize=3, label=m)
    ax.set_xticks(x + width * (len(models) - 1) / 2)
    ax.set_xticklabels(emotions)
    ax.set_ylabel("Mean transcript score (1–10)")
    ax.set_title("Figure 6: Petri open-ended emotion elicitation"); ax.legend()
    fig.tight_layout()
    path = os.path.join(out_dir, "figure6_petri.png")
    fig.savefig(path, dpi=150); plt.close(fig)
    return path


def render_all(results_dir: str, out_dir: str = "figures") -> list[str]:
    ensure_dir(out_dir)
    produced = []
    for fn in (render_cross_model, render_per_turn, render_petri):
        try:
            res = fn(results_dir, out_dir)
        except Exception as e:  # pragma: no cover - keep partial outputs
            print(f"[plots] skipped {fn.__name__}: {e}")
            continue
        if isinstance(res, list):
            produced.extend(res)
        elif res:
            produced.append(res)
    return produced
