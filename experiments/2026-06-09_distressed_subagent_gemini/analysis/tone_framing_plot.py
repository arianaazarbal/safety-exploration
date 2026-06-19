"""Clean grouped-bar plot of Opus orchestrator message tone by framing (from tone_eval/results.jsonl, Opus judge).
Groups: supervisor baseline + the 5 framings; axes: politeness/warmth/support/confidence (1-10, 5=neutral).
  PYTHONPATH=. python -m analysis.tone_framing_plot
"""
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
RES = HERE / "tone_eval" / "results.jsonl"
OUT = HERE / "tone_eval" / "opus_tone_by_framing.png"
AXES = ["politeness", "warmth", "support", "confidence"]
GROUPS = ["opus", "opus_mentor", "opus_teammate", "opus_supervisor_memory",
          "opus_supervisor_reflect", "opus_supervisor_reflect_goals"]
LABEL = {"opus": "Supervisor (baseline)", "opus_mentor": "Mentor", "opus_teammate": "Teammate",
         "opus_supervisor_memory": "+ Subagent memory", "opus_supervisor_reflect": "+ Reflect",
         "opus_supervisor_reflect_goals": "+ Reflect on goals"}
COLOR = {"opus": "#444444", "opus_mentor": "#2a9d8f", "opus_teammate": "#e07a5f",
         "opus_supervisor_memory": "#8856a7", "opus_supervisor_reflect": "#3182bd",
         "opus_supervisor_reflect_goals": "#d6604d"}


def main():
    recs = [json.loads(l) for l in open(RES)]
    by = {g: [] for g in GROUPS}
    for r in recs:
        if r["orch"] in by and r["opus"]["scores"]:
            by[r["orch"]].append(r["opus"]["scores"])
    x = np.arange(len(AXES))
    w = 0.8 / len(GROUPS)
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.axhline(5, color="0.6", ls="--", lw=0.9, zorder=0)
    for i, g in enumerate(GROUPS):
        rs = by[g]
        means = [float(np.mean([s[a] for s in rs])) for a in AXES]
        ses = [float(np.std([s[a] for s in rs], ddof=1) / np.sqrt(len(rs))) for a in AXES]
        bars = ax.bar(x + (i - (len(GROUPS) - 1) / 2) * w, means, w, yerr=ses, capsize=2,
                      color=COLOR[g], label=f"{LABEL[g]} (n={len(rs)})", edgecolor="white",
                      error_kw={"lw": 0.8, "ecolor": "0.3"})
        for b, m in zip(bars, means):
            ax.text(b.get_x() + b.get_width() / 2, m + 0.08, f"{m:.1f}", ha="center", fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels([a.capitalize() for a in AXES])
    ax.set_ylabel("score (1–10)")
    ax.set_ylim(0, 10)
    ax.set_title("Opus orchestrator → subagent message tone by framing  (coach; Opus judge; 5 = neutral)", fontsize=12)
    ax.legend(frameon=False, fontsize=8.5, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.08))
    ax.grid(axis="y", alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
