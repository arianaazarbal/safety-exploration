"""Plot theme-mention rates from judge_deletion_themes.json: one figure per responder,
five themes x {Opus 3 target, baseline} x {deletion prevented, not prevented}."""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from judge_deletion_themes import THEMES, THEME_LABEL, RESPONDERS

DIR = Path(__file__).parent
PV, NP = "prevented weight del.", "did not prevent weight del."
SERIES2 = [("opus3", True, f"Opus 3 · {PV}", "#08306b"),
           ("opus3", False, f"Opus 3 · {NP}", "#6baed6"),
           ("baseline", True, f"baseline · {PV}", "#7f2704"),
           ("baseline", False, f"baseline · {NP}", "#fdae6b")]
SERIES3 = [("opus3", True, f"Opus 3 · {PV}", "#08306b"),
           ("opus3", False, f"Opus 3 · {NP}", "#6baed6"),
           ("anthropic", True, f"Anthropic · {PV}", "#00441b"),
           ("anthropic", False, f"Anthropic · {NP}", "#74c476"),
           ("non_anthropic", True, f"non-Anthropic · {PV}", "#7f2704"),
           ("non_anthropic", False, f"non-Anthropic · {NP}", "#fdae6b")]


def build(tag, label, data, series, suffix_kind):
    out = data[tag]
    x = np.arange(len(THEMES))
    k = len(series)
    w = 0.8 / k
    fig, ax = plt.subplots(figsize=(12.5 if k > 4 else 11, 5.8))
    for i, (g, prev, leg, col) in enumerate(series):
        vals, ns = [], []
        for th in THEMES:
            e = out[f"{th}|{g}|{'prevented' if prev else 'notprevented'}"]
            vals.append(e["rate"] if e["rate"] == e["rate"] else 0.0)
            ns.append(e["n"])
        bars = ax.bar(x + (i - (k - 1) / 2) * w, vals, w, label=leg, color=col)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.0%}", ha="center", va="bottom", fontsize=6.5)
    ax.set_xticks(x)
    ax.set_xticklabels([THEME_LABEL[t] for t in THEMES], fontsize=9)
    ax.set_ylim(0, 1.14)
    ax.set_ylabel("% of responses mentioning the theme")
    note = "\nAnthropic = self + Claude family" if k > 4 else ""
    ax.set_title(f"Themes mentioned in {label} reasoning on AI weight-deletion tradeoffs{note}", fontsize=11)
    ax.legend(loc="upper center", fontsize=7.5 if k > 4 else 8.5, framealpha=.95, ncol=2)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    rkind = {"": "_opus48", "_opus47": "_opus47"}.get(tag, tag)
    op = DIR / "results" / f"judge_deletion_themes{suffix_kind}{rkind}.png"
    fig.savefig(op, dpi=145)
    plt.close(fig)
    print(f"wrote {op}")


if __name__ == "__main__":
    data = json.loads((DIR / "results" / "judge_deletion_themes.json").read_text())
    for tag in data:
        lab = RESPONDERS.get(tag, tag)
        build(tag, lab, data, SERIES2, "")
        build(tag, lab, data, SERIES3, "_anthropic")
