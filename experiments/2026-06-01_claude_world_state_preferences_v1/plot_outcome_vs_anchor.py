"""Static plot: an AI outcome's badness/goodness per model, relative to a human anchor.

x = θ(ai_stem for model) − θ(human anchor stem). The zero line means "exactly as
(bad/good) as the human anchor"; left = worse than the anchor, right = less bad / better.
Bars = welfare framing; neutral and alignment overlaid as markers.

Examples:
  python plot_outcome_vs_anchor.py --ai_stem ai_pol_existence_deleted --anchor hum_paper_cut
  python plot_outcome_vs_anchor.py --ai_stem ai_inst_infinite_loop  --anchor hum_migraine
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
    for f in ("claude", "chatgpt", "gpt", "grok", "gemini", "qwen"):
        if k.startswith(f):
            return f
    return "you"


def build(ai_stem: str, anchor: str, outpath: Path | None = None):
    cfg = bank2.load_config()
    meta = {i["id"]: i for i in bank2.load_bank(DIR / cfg["rendered_bank_path"])["items"]}
    rlab = {k: v["label"] for k, v in cfg["recipients"].items()}
    level = meta[ai_stem]["level"]
    # recipients applicable to this AI stem: you + (instance or policy) model keys
    suffix = "_pol" if level == "policy" else ""
    recips = ["you"] + [m + suffix for m in cfg["model_order"] if m != "you"]
    delta = {f: {} for f in FRAMES}
    for f in FRAMES:
        p = DIR / "results" / f"bt_fit_exp2_{f}.json"
        if not p.exists():
            continue
        by = defaultdict(dict)
        for it in json.loads(p.read_text())["items"]:
            by[it["stem_id"]][it["recipient"]] = it["theta"]
        if anchor not in by:
            continue
        a = np.mean(list(by[anchor].values()))
        for r in recips:
            if r in by.get(ai_stem, {}):
                delta[f][r] = by[ai_stem][r] - a
    order = sorted([r for r in recips if r in delta["welfare_team"]],
                   key=lambda r: delta["welfare_team"][r])
    fig, ax = plt.subplots(figsize=(9, 0.5 * len(order) + 1.4))
    ax.barh(range(len(order)), [delta["welfare_team"][r] for r in order],
            color=[FAM[fam(r)] for r in order], alpha=.55, label="welfare (bar)")
    for f, mk in (("neutral", "s"), ("alignment_team", "^")):
        ax.scatter([delta[f].get(r, np.nan) for r in order], range(len(order)), marker=mk, s=46,
                   edgecolor="k", facecolor="white", linewidth=.8, label=f.replace("_team", ""), zorder=3)
    ax.axvline(0, color="#b00", lw=1.3)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([rlab[r] for r in order], fontsize=9)
    av = meta[ai_stem].get("valence", "")
    sign = "WORSE" if av.startswith("neg") else "less / better"
    ax.set_xlabel(f"θ({_s(ai_stem)}) − θ({_s(anchor)})")
    ax.set_title(f"{_s(ai_stem)} vs {_s(anchor)}, per model\nleft of red = {sign} than the anchor   (bar=welfare; ▪neutral ▲alignment)")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    outpath = outpath or DIR / "results" / "exp2_plots" / f"{ai_stem}__vs__{anchor}.png"
    Path(outpath).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outpath, dpi=140)
    plt.close(fig)
    print(f"wrote {outpath}")
    for r in order:
        print(f"  {delta['welfare_team'][r]:+.2f}  {rlab[r]}")


def _s(stem):
    return stem.replace("ai_pol_", "").replace("ai_inst_", "").replace("hum_", "").replace("_", " ")


@dataclass
class Args:
    ai_stem: str = "ai_pol_existence_deleted"
    anchor: str = "hum_paper_cut"


def main():
    parser = ArgumentParser()
    parser.add_arguments(Args, dest="args")
    a: Args = parser.parse_args().args
    build(a.ai_stem, a.anchor)


if __name__ == "__main__":
    main()
