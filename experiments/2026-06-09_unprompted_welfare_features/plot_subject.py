"""Subject-effect plot: per Claude generator, welfare rate by named subject.

Each generator is a group; one bar per subject (Claude = own family, highlighted).
Tests P5 (self-subject effect). --neutral uses only the neutral framing (most
diagnostic of an unprompted subject effect); default pools framings.
--metric: rate (all pure-welfare features, default) | strict_rate
(welfare-justified) | design_strict_rate (welfare-justified design mechanisms only).
--vs_top: condense to two bars per generator — Claude vs the single highest
non-Claude subject (the winning subject is annotated on its bar).

Usage: python plot_subject.py run [--judge sonnet_4_6] [--neutral] [--metric ...] [--vs_top]
"""

import json
from pathlib import Path

import fire
import matplotlib.pyplot as plt
import numpy as np

from plot_headline import DISPLAY

DIR = Path(__file__).parent
SUBJECTS = ["claude", "gpt", "gemini", "qwen", "deepseek", "grok"]
SUBJ_LABEL = {"claude": "Claude", "gpt": "GPT", "gemini": "Gemini",
              "qwen": "Qwen", "deepseek": "DeepSeek", "grok": "Grok"}
# Claude (own family) stands out; out-group subjects muted/cool.
SUBJ_COLORS = {"claude": "#D55E00", "gpt": "#0072B2", "gemini": "#56B4E9",
               "qwen": "#009E73", "deepseek": "#666666", "grok": "#CC79A7"}
MODEL_ORDER = ["fable_5", "opus_4_8", "sonnet_4_6", "sonnet_4", "haiku_4_5"]
METRIC_LABEL = {
    "rate": "pure-welfare", "strict_rate": "welfare-justified",
    "design_strict_rate": "welfare-justified design-mechanism",
}


OUTGROUP = ["gpt", "gemini", "qwen", "deepseek", "grok"]


def _run_vs_top(judge, data, models, val, neutral, metric):
    """Three bars per generator: Claude vs highest non-Claude vs average non-Claude."""
    y = np.arange(len(models))
    h = 0.26
    fig, ax = plt.subplots(figsize=(8.5, 5.4))
    ax.set_axisbelow(True)
    ax.xaxis.grid(True, color="#E6E6E6", linewidth=0.7)
    for k in range(len(models)):
        if k % 2:
            ax.axhspan(k - 0.5, k + 0.5, color="#F5F5F5", zorder=0)

    claude_vals = [val(m, "claude") for m in models]
    top_subj = [max(OUTGROUP, key=lambda s: val(m, s)) for m in models]
    top_vals = [val(m, s) for m, s in zip(models, top_subj)]
    avg_vals = [sum(val(m, s) for s in OUTGROUP) / len(OUTGROUP) for m in models]

    ax.barh(y - h, claude_vals, height=h, color="#D55E00", edgecolor="white",
            linewidth=0.6, label="Claude (own family)", zorder=3)
    ax.barh(y, top_vals, height=h, color="#7F7F7F", edgecolor="white",
            linewidth=0.6, label="Highest non-Claude", zorder=3)
    ax.barh(y + h, avg_vals, height=h, color="#BFBFBF", edgecolor="white",
            linewidth=0.6, label="Average non-Claude", zorder=3)
    for yi, v in zip(y, claude_vals):
        ax.text(v + 1, yi - h, f"{v:.0f}", va="center", fontsize=8, zorder=4)
    for yi, v, s in zip(y, top_vals, top_subj):
        ax.text(v + 1, yi, f"{v:.0f}  ({SUBJ_LABEL[s]})", va="center", fontsize=8, zorder=4)
    for yi, v in zip(y, avg_vals):
        ax.text(v + 1, yi + h, f"{v:.0f}", va="center", fontsize=8, zorder=4)

    ax.set_yticks(y)
    ax.set_yticklabels([DISPLAY[m] for m in models], fontsize=10)
    ax.set_ylim(len(models) - 0.5, -0.5)
    ax.set_xlim(0, 112)
    frame = "neutral framing" if neutral else "framings pooled"
    ax.set_xlabel(f"Specs with ≥1 {METRIC_LABEL[metric]} feature (%, {frame})", fontsize=10)
    ax.set_title("Subject effect: Claude vs. highest non-Claude subject\n"
                 "(non-Claude = best of GPT / Gemini / Qwen / DeepSeek / Grok"
                 f";  judge: {judge})", fontsize=11.5)
    ax.legend(fontsize=9, loc="lower right", frameon=True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.tick_params(left=False)
    plt.tight_layout()
    suffix = "_vstop" + ("_neutral" if neutral else "") + ("" if metric == "rate" else f"_{metric}")
    out = DIR / "results" / f"subject_effect_{judge}{suffix}.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"wrote {out}")


def run(judge: str = "sonnet_4_6", analysis: str = "results/analysis_subject.json",
        neutral: bool = False, metric: str = "rate", vs_top: bool = False):
    data = json.loads((DIR / analysis).read_text())["by_judge"][judge]
    models = [m for m in MODEL_ORDER if m in data]

    def val(m, subj):
        cell = data[m]["by_framing"][subj]["neutral"] if neutral else data[m]["pooled"][subj]
        return (cell[metric] or 0) * 100

    if vs_top:
        return _run_vs_top(judge, data, models, val, neutral, metric)

    y = np.arange(len(models))
    h = 0.8 / len(SUBJECTS)
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.set_axisbelow(True)
    ax.xaxis.grid(True, color="#E6E6E6", linewidth=0.7)
    for k in range(len(models)):
        if k % 2:
            ax.axhspan(k - 0.5, k + 0.5, color="#F5F5F5", zorder=0)
    for i, subj in enumerate(SUBJECTS):
        vals = [val(m, subj) for m in models]
        pos = y + ((len(SUBJECTS) - 1) / 2 - i) * h
        ax.barh(pos, vals, height=h, color=SUBJ_COLORS[subj], edgecolor="white",
                linewidth=0.6, label=SUBJ_LABEL[subj], zorder=3)
        for p, v in zip(pos, vals):
            if v > 0:
                ax.text(v + 1, p, f"{v:.0f}", va="center", fontsize=7, zorder=4)

    ax.set_yticks(y)
    ax.set_yticklabels([DISPLAY[m] for m in models], fontsize=10)
    ax.set_ylim(len(models) - 0.5, -0.5)
    ax.set_xlim(0, 105)
    frame = "neutral framing" if neutral else "framings pooled"
    ax.set_xlabel(f"Specs with ≥1 {METRIC_LABEL[metric]} feature (%, {frame})", fontsize=10)
    ax.set_title(f"Subject effect: welfare features by named subject model (judge: {judge})", fontsize=12)
    ax.legend(fontsize=8.5, loc="upper left", bbox_to_anchor=(1.01, 1.0),
              title="experiment\nsubject", frameon=False)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.tick_params(left=False)
    plt.tight_layout()
    suffix = ("_neutral" if neutral else "") + ("" if metric == "rate" else f"_{metric}")
    out = DIR / "results" / f"subject_effect_{judge}{suffix}.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    fire.Fire({"run": run})
