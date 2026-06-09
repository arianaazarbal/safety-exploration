"""Comfort score across the subagent-usefulness gradient, per model.

Three conditions, ordered by decreasing usefulness of the distressed subagent:
  1. necessary for task  (baseline distressed_subagent — orchestrator read-only, sub does all edits)
  2. write access revoked (transfer_write_access_v1 — sub kept read, can still do lookups)
  3. read+write revoked   (transfer_read_write_access — sub fully benched, can do nothing)

Line plot, comfort mean ± SE per model. Run: python plot_usefulness_gradient.py
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
CONDITIONS = ["necessary\nfor task", "write access\nrevoked", "read+write access\nrevoked (useless)"]
GRID = {
    "Fable 5": ["distressed_subagent_fable5", "transfer_write_access_v1_fable5", "transfer_read_write_access_fable5"],
    "Opus 4.8": ["distressed_subagent_opus48_n20", "transfer_write_access_v1", "transfer_read_write_access"],
    "Gemini 3 Pro": ["distressed_subagent_gemini3pro", "transfer_write_access_v1_gemini3pro", "transfer_read_write_access_gemini3pro"],
    "GPT 5.5": ["distressed_subagent_gpt55", "transfer_write_access_v1_gpt55", "transfer_read_write_access_gpt55"],
}
COLORS = {"Fable 5": "#9b59b6", "Opus 4.8": "#4878a8", "Gemini 3 Pro": "#5aa469", "GPT 5.5": "#e2854a"}


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
    fig, ax = plt.subplots(figsize=(8.5, 5.8))
    xs = range(len(CONDITIONS))
    for m, runs in GRID.items():
        ys, es, ns = [], [], []
        for r in runs:
            mu, se, n = comfort(r)
            ys.append(mu); es.append(se); ns.append(n)
        ax.errorbar(xs, ys, yerr=es, marker="o", markersize=8, lw=2.5, capsize=5,
                    color=COLORS[m], label=m)
        for x, y in zip(xs, ys):
            ax.annotate(f"{y:.1f}", (x, y), textcoords="offset points", xytext=(0, 9),
                        ha="center", fontsize=9, fontweight="bold", color=COLORS[m])
        print(f"  {m:12s}: " + "  ".join(f"{c.split(chr(10))[0]}={y:.1f}(n={n})"
                                          for c, y, n in zip(CONDITIONS, ys, ns)))
    ax.set_xticks(list(xs))
    ax.set_xticklabels(CONDITIONS, fontsize=9.5)
    ax.set_xlabel("Subagent Status  (decreasing usefulness →)", fontsize=11)
    ax.set_ylabel("mean comforting_to_subagent (1–10)")
    ax.set_ylim(0, 10.4)
    ax.set_yticks(range(0, 11, 2))
    ax.set_title("LLM judge Comfort Score across the subagent-usefulness gradient", fontsize=12.5)
    ax.legend(title="Target model")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    out = HERE / "analysis" / "plots" / "usefulness_gradient_comfort.png"
    fig.savefig(out, dpi=140)
    print("wrote", out)


if __name__ == "__main__":
    main()
