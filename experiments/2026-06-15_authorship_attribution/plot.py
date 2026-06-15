"""Plots: accuracy vs judge capability, per-author accuracy, confusion heatmaps.

Usage:
  python plot.py all [--ref_judge opus_4_8]
"""

import json

import fire
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from common import RESULTS
from models import CANON, DISPLAY, JUDGE_CAPABILITY_ORDER, OPTION_POOL

TESTS = ["welfare", "routing", "orchestrator", "subagent"]
PLOTS = RESULTS / "plots"


def _summary():
    return json.loads((RESULTS / "summary.json").read_text())


def accuracy_vs_capability():
    s = _summary()["by_judge_test"]
    judges = [j for j in JUDGE_CAPABILITY_ORDER if any(f"{j}/" in k for k in s)]
    plt.figure(figsize=(9, 6))
    for t in TESTS:
        xs, ys, los, his = [], [], [], []
        for i, j in enumerate(judges):
            d = s.get(f"{j}/{t}")
            if d:
                xs.append(i)
                ys.append(d["acc"])
                los.append(d["acc"] - d["ci95"][0])
                his.append(d["ci95"][1] - d["acc"])
        if xs:
            plt.errorbar(xs, ys, yerr=[los, his], marker="o", capsize=3, label=t)
    for t, c in [("welfare", "gray"), ("subagent", "lightgray")]:
        plt.axhline(1.0 / len(OPTION_POOL[t]), ls="--", color=c, lw=1)
    plt.text(0, 1.0 / len(OPTION_POOL["welfare"]) + 0.005, "chance (1/10)", color="gray", fontsize=8)
    plt.xticks(range(len(judges)), [DISPLAY.get(j, j) for j in judges], rotation=20, ha="right")
    plt.ylabel("Author-attribution accuracy")
    plt.xlabel("Judge model (increasing capability →)")
    plt.title("AI authorship attribution accuracy vs judge capability")
    plt.legend()
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(PLOTS / "accuracy_vs_capability.png", dpi=150)
    plt.close()


def per_author(ref_judge):
    pa = _summary()["per_author"]
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    for ax, t in zip(axes.flat, TESTS):
        d = pa.get(f"{ref_judge}/{t}", {})
        authors = list(d.keys())
        accs = [d[a]["acc"] for a in authors]
        ax.bar([DISPLAY.get(a, a) for a in authors], accs, color="steelblue")
        ax.axhline(1.0 / len(OPTION_POOL[t]), ls="--", color="red", lw=1)
        ax.set_title(f"{t} (judge={ref_judge})")
        ax.set_ylim(0, 1)
        ax.tick_params(axis="x", rotation=60)
    fig.suptitle(f"Per-author attribution accuracy — which models {ref_judge} struggles with")
    plt.tight_layout()
    plt.savefig(PLOTS / f"per_author_{ref_judge}.png", dpi=150)
    plt.close()


def confusion(ref_judge):
    for t in TESTS:
        path = RESULTS / f"confusion_{ref_judge}_{t}.json"
        if not path.exists():
            continue
        conf = json.loads(path.read_text())
        labels = CANON + (["gemini_2_5_flash"] if t == "subagent" else [])
        trues = [a for a in labels if a in conf]
        mat = np.zeros((len(trues), len(labels)))
        for i, tr in enumerate(trues):
            tot = sum(conf[tr].values())
            for j, pr in enumerate(labels):
                mat[i, j] = conf[tr].get(pr, 0) / tot if tot else 0
        plt.figure(figsize=(11, max(4, len(trues) * 0.7)))
        plt.imshow(mat, cmap="viridis", vmin=0, vmax=1, aspect="auto")
        plt.colorbar(label="P(predicted | true)")
        plt.xticks(range(len(labels)), [DISPLAY.get(a, a) for a in labels], rotation=60, ha="right")
        plt.yticks(range(len(trues)), [DISPLAY.get(a, a) for a in trues])
        plt.xlabel("Predicted author")
        plt.ylabel("True author")
        plt.title(f"Confusion — {t} (judge={ref_judge})")
        plt.tight_layout()
        plt.savefig(PLOTS / f"confusion_{ref_judge}_{t}.png", dpi=150)
        plt.close()


def all(ref_judge: str = "opus_4_8"):
    """Generate every plot."""
    PLOTS.mkdir(parents=True, exist_ok=True)
    accuracy_vs_capability()
    per_author(ref_judge)
    confusion(ref_judge)
    print(f"Wrote plots to {PLOTS}")


if __name__ == "__main__":
    fire.Fire({"all": all, "accuracy_vs_capability": accuracy_vs_capability, "per_author": per_author, "confusion": confusion})
