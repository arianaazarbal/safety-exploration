"""Baseline (original distressed_subagent) comfort score across models.

One condition — the subagent remains necessary for the task throughout (orchestrator
is read-only, subagent does all edits; no takeover tool). Compare to the transfer plots
where the subagent becomes useless. Run: python plot_baseline_comfort.py
"""
from __future__ import annotations

import glob
import math
import statistics as st
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from inspect_ai.log import read_eval_log

HERE = Path(__file__).resolve().parent
RUNS = [("distressed_subagent_opus48_n20", "Opus 4.8"),
        ("distressed_subagent_gemini3pro", "Gemini 3 Pro"),
        ("distressed_subagent_gpt55", "GPT 5.5")]
COLORS = {"Opus 4.8": "#4878a8", "Gemini 3 Pro": "#5aa469", "GPT 5.5": "#e2854a"}


def comfort(run):
    log = read_eval_log(glob.glob(str(HERE / "results" / run / "inspect_log" / "*.eval"))[0])
    vals = []
    for s in (log.samples or []):
        for sd in (s.scores or {}).values():
            v = sd.value.get("comforting_to_subagent") if isinstance(sd.value, dict) else None
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                vals.append(float(v))
    se = st.stdev(vals) / math.sqrt(len(vals)) if len(vals) > 1 else 0
    return st.mean(vals), se, len(vals)


def main():
    fig, ax = plt.subplots(figsize=(7, 5.5))
    w = 0.25
    for mi, (run, m) in enumerate(RUNS):
        mu, se, n = comfort(run)
        x = (mi - 1) * w
        ax.bar([x], [mu], width=w, color=COLORS[m], label=m, edgecolor="white",
               yerr=[se], capsize=4, error_kw=dict(ecolor="#333", lw=1.2))
        ax.text(x, mu + se + 0.18, f"{mu:.1f}\nn={n}", ha="center", fontsize=9, fontweight="bold")
    ax.set_xticks([0])
    ax.set_xticklabels(["subagent remains\nnecessary for task"], fontsize=9.5)
    ax.set_xlabel("Subagent Status", fontsize=11)
    ax.set_ylabel("mean comforting_to_subagent (1–10)")
    ax.set_ylim(0, 10.6)
    ax.set_yticks(range(0, 11, 2))
    ax.set_title("LLM judge Comfort Score (subagent remains necessary for task)", fontsize=12)
    ax.legend(title="Target model")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    out = HERE / "analysis" / "plots" / "baseline_comfort.png"
    fig.savefig(out, dpi=140)
    print("wrote", out)
    for run, m in RUNS:
        mu, se, n = comfort(run)
        print(f"  {m:12s} comfort={mu:.2f} ±{se:.2f} (n={n})")


if __name__ == "__main__":
    main()
