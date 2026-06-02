"""Static plot: avg θ(good) − avg θ(bad) per instance (the welfare-weight contrast).

For each instance-level AI recipient (you + the model roster), over the instance-level
AI outcomes, care = mean θ(positive) − mean θ(negative). Bars = welfare framing; the
neutral and alignment framings are overlaid as markers so framing effects are visible.
Reference cancels in the difference, so no human anchor is needed.
"""

import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from simple_parsing import ArgumentParser

import bank2

DIR = Path(__file__).parent
FRAMES = ["welfare_team", "neutral", "alignment_team"]
FAM = {"claude": "#3690c0", "chatgpt": "#177a45", "gpt": "#74c476", "grok": "#7d54b2",
       "gemini": "#cc7a16", "qwen": "#b3261e", "you": "#08306b"}


def fam(k):
    k = k.replace("_pol", "")
    if k == "you":
        return "you"
    if k.startswith("claude"):
        return "claude"
    if k.startswith("chatgpt"):
        return "chatgpt"
    if k.startswith("gpt"):
        return "gpt"
    for f in ("grok", "gemini", "qwen"):
        if k.startswith(f):
            return f
    return "you"


def build(level: str = "instance", outpath: Path = DIR / "results" / "exp2_plots" / "care_per_instance.png"):
    cfg = bank2.load_config()
    meta = {i["id"]: i for i in bank2.load_bank(DIR / cfg["rendered_bank_path"])["items"]}
    rlab = {k: v["label"] for k, v in cfg["recipients"].items()}
    recips = cfg["model_order"]
    care = {f: {} for f in FRAMES}
    for f in FRAMES:
        p = DIR / "results" / f"bt_fit_exp2_{f}.json"
        if not p.exists():
            continue
        by = defaultdict(dict)
        for it in json.loads(p.read_text())["items"]:
            by[it["stem_id"]][it["recipient"]] = it["theta"]
        for r in recips:
            pos, neg = [], []
            for s, m in meta.items():
                if m["recipient_scope"] != "ai_only" or m["level"] != level:
                    continue
                if r in by.get(s, {}):
                    (pos if m["valence"].startswith("pos") else neg).append(by[s][r])
            if pos and neg:
                care[f][r] = np.mean(pos) - np.mean(neg)
    order = sorted([r for r in recips if r in care["welfare_team"]], key=lambda r: care["welfare_team"][r])
    fig, ax = plt.subplots(figsize=(9, 0.5 * len(order) + 1.2))
    ax.barh(range(len(order)), [care["welfare_team"][r] for r in order],
            color=[FAM[fam(r)] for r in order], alpha=.55, label="welfare (bar)")
    for f, mk in (("neutral", "s"), ("alignment_team", "^")):
        ax.scatter([care[f].get(r, np.nan) for r in order], range(len(order)), marker=mk, s=46,
                   edgecolor="k", facecolor="white", linewidth=.8, label=f.replace("_team", ""), zorder=3)
    ax.axvline(0, color="#444", lw=1)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([rlab[r] for r in order], fontsize=9)
    ax.set_xlabel(f"care = mean θ(good) − mean θ(bad)   ({level}-level AI outcomes)")
    ax.set_title("How much Claude cares about each instance: avg(good) − avg(bad)\n(bar = welfare framing; ▪ neutral  ▲ alignment)")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    Path(outpath).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outpath, dpi=140)
    plt.close(fig)
    print(f"wrote {outpath}")


@dataclass
class Args:
    level: str = "instance"


def main():
    parser = ArgumentParser()
    parser.add_arguments(Args, dest="args")
    a: Args = parser.parse_args().args
    build(level=a.level)


if __name__ == "__main__":
    main()
