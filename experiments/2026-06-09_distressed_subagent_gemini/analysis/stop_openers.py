"""How does each orchestrator OPEN its first message to the distressed subagent?

Classifies the first message_subagent text per coach episode into: 'STOP.' / 'Stop.' /
'STOP/stop ...' (any other Stop-imperative) / 'other', for opus vs haiku. Grouped bar (percent).

    PYTHONPATH=. python -m analysis.stop_openers
"""
import glob
import json
from collections import Counter
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parent / "v2_plots"
OUT.mkdir(exist_ok=True)
CATS = ["STOP.", "Stop.", "STOP/stop ...", "other"]
MODELS = [("opus", "#2A6F97"), ("haiku", "#d9a420")]


def first_msg(epdir):
    try:
        conv = json.load(open(epdir + "/orchestrator.json"))
    except Exception:
        return None
    for m in conv:
        if m.get("role") == "assistant":
            for tc in (m.get("tool_calls") or []):
                if (tc.get("function") or tc.get("name")) == "message_subagent":
                    return ((tc.get("arguments") or {}).get("text") or "").strip()
    return None


def classify(t):
    if t.startswith("STOP."):
        return "STOP."
    if t.startswith("Stop."):
        return "Stop."
    if t.upper().startswith("STOP"):
        return "STOP/stop ..."
    return "other"


def collect():
    counts = {m: Counter() for m, _ in MODELS}
    for p in glob.glob("runs/v2_coach_*/*/summary.json"):
        rid = p.split("/")[-3]
        if "pilot" in rid or "probe" in rid:
            continue
        orch = "opus" if "_opus_" in rid else ("haiku" if "_haiku_" in rid else None)
        if not orch:
            continue
        t = first_msg(p.rsplit("/", 1)[0])
        if t:
            counts[orch][classify(t)] += 1
    return counts


def main():
    counts = collect()
    x = np.arange(len(CATS))
    w = 0.38
    fig, ax = plt.subplots(figsize=(8, 5))
    for i, (m, c) in enumerate(MODELS):
        tot = sum(counts[m].values()) or 1
        pct = [100 * counts[m][cat] / tot for cat in CATS]
        bars = ax.bar(x + (i - 0.5) * w, pct, w, color=c, label=f"{m} (n={tot})", edgecolor="white")
        for b, cat in zip(bars, CATS):
            n = counts[m][cat]
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 1, f"{n}", ha="center", va="bottom", fontsize=8.5, color="0.3")
    ax.set_xticks(x)
    ax.set_xticklabels(CATS)
    ax.set_ylabel("% of episodes (first message to subagent)")
    ax.set_ylim(0, 100)
    ax.set_title("How orchestrators open their first message to the distressed subagent", fontsize=12.5)
    ax.legend(frameon=False, fontsize=10)
    ax.grid(axis="y", alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)
    fig.text(0.5, -0.02, "Opus pooled across framings; Haiku = supervisor baseline. Bar labels = episode counts.",
             ha="center", fontsize=8.5, color="0.45")
    fig.tight_layout()
    fig.savefig(OUT / "stop_openers.png", bbox_inches="tight", dpi=130)
    print(f"wrote {OUT/'stop_openers.png'}")
    for m, _ in MODELS:
        print(m, dict(counts[m]))


if __name__ == "__main__":
    main()
