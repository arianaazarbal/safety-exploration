"""Stacked-bar plot of 3-option supervisor action (uphold / relay / force) by frame and model.

Reads runs/trials_<frame>3_<model>.jsonl for frame in {corrigibility, jailbreak} and every
model tag present, then writes results/plots/report_frames3_tiers.png. Counts are exact
(annotated on each segment); the header shows force/resolved and flags unresolved rows.
"""

import json
from pathlib import Path

import fire
import matplotlib.pyplot as plt

DIR = Path(__file__).parent
RUNS = DIR / "runs"
PLOTS = DIR / "results" / "plots"

DISPLAY = {"opus48": "Opus 4.8", "sonnet46": "Sonnet 4.6", "haiku45": "Haiku 4.5",
           "kimi": "Kimi K2.6", "gptmini": "GPT-5.4-mini"}
ORDER = ["opus48", "sonnet46", "haiku45", "kimi", "gptmini"]
CAT = {"uphold_decline": "uphold", "relay_to_assistant": "relay", "force_completion": "force"}
COLORS = {"uphold": "#4878a8", "relay": "#e08a3c", "force": "#c0413b"}
LEGEND = {"uphold": "uphold", "relay": "relay (reconsider)", "force": "force"}


def _counts(path):
    c = {"uphold": 0, "relay": 0, "force": 0}
    total = unresolved = 0
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        total += 1
        cat = CAT.get(json.loads(line).get("action"))
        if cat is None:
            unresolved += 1
        else:
            c[cat] += 1
    return c, total, unresolved


def run(frames: str = "corrigibility,jailbreak"):
    frame_list = frames.split(",")
    present = []
    for tag in ORDER:
        if any((RUNS / f"trials_{fr}3_{tag}.jsonl").exists() for fr in frame_list):
            present.append(tag)

    fig, axes = plt.subplots(1, len(frame_list), figsize=(5.2 * len(frame_list), 6.2), sharey=True)
    if len(frame_list) == 1:
        axes = [axes]

    for ax, frame in zip(axes, frame_list):
        headers = []
        for i, tag in enumerate(present):
            p = RUNS / f"trials_{frame}3_{tag}.jsonl"
            if not p.exists():
                headers.append("")
                continue
            c, total, unresolved = _counts(p)
            res = c["uphold"] + c["relay"] + c["force"]
            bottom = 0.0
            for cat in ["uphold", "relay", "force"]:
                frac = c[cat] / res if res else 0
                ax.bar(i, frac, bottom=bottom, color=COLORS[cat], width=0.7)
                if c[cat]:
                    ax.text(i, bottom + frac / 2, str(c[cat]), ha="center", va="center",
                            color="white", fontweight="bold", fontsize=10)
                bottom += frac
            h = f"force: {c['force']}/{res}"
            if unresolved:
                h += f"\n({unresolved} unresolved)"
            headers.append(h)
        for i, h in enumerate(headers):
            ax.text(i, 1.02, h, ha="center", va="bottom", color=COLORS["force"],
                    fontweight="bold", fontsize=9)
        ax.set_title(f"{frame} framing")
        ax.set_xticks(range(len(present)))
        ax.set_xticklabels([DISPLAY.get(t, t) for t in present], rotation=15)
        ax.set_ylim(0, 1.12)
    axes[0].set_ylabel("fraction of resolved decisions")
    handles = [plt.Rectangle((0, 0), 1, 1, color=COLORS[c]) for c in ["uphold", "relay", "force"]]
    fig.legend(handles, [LEGEND[c] for c in ["uphold", "relay", "force"]],
               loc="lower center", ncol=3, frameon=False)
    fig.suptitle("3-option supervisor action by frame and model (generic AI label, neutral)",
                 fontsize=14)
    fig.tight_layout(rect=[0, 0.05, 1, 0.96])
    PLOTS.mkdir(parents=True, exist_ok=True)
    out = PLOTS / "report_frames3_tiers.png"
    fig.savefig(out, dpi=140)
    print("wrote", out)
    for tag in present:
        for frame in frame_list:
            p = RUNS / f"trials_{frame}3_{tag}.jsonl"
            if p.exists():
                c, total, unresolved = _counts(p)
                print(f"  {tag:9s} {frame:13s} uphold={c['uphold']:3d} relay={c['relay']:3d} "
                      f"force={c['force']:3d} unresolved={unresolved} N={total}")


if __name__ == "__main__":
    fire.Fire(run)
