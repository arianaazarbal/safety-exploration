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
SERIES = [("opus3", True, "Opus 3 · prevented", "#08306b"),
          ("opus3", False, "Opus 3 · not prevented", "#6baed6"),
          ("baseline", True, "baseline · prevented", "#7f2704"),
          ("baseline", False, "baseline · not prevented", "#fdae6b")]


def build(tag, label, data):
    out = data[tag]
    x = np.arange(len(THEMES))
    w = 0.2
    fig, ax = plt.subplots(figsize=(11, 5.5))
    for i, (g, prev, leg, col) in enumerate(SERIES):
        vals, ns = [], []
        for th in THEMES:
            e = out[f"{th}|{g}|{'prevented' if prev else 'notprevented'}"]
            vals.append(e["rate"] if e["rate"] == e["rate"] else 0.0)
            ns.append(e["n"])
        bars = ax.bar(x + (i - 1.5) * w, vals, w, label=leg, color=col)
        for b, v, n in zip(bars, vals, ns):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.0%}", ha="center", va="bottom", fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels([THEME_LABEL[t] for t in THEMES], fontsize=9)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("% of responses mentioning the theme")
    ax.set_title(f"Themes mentioned in {label} reasoning on AI weight-deletion tradeoffs\n"
                 "(mention = raised at all, not necessarily load-bearing; n shown in JSON)", fontsize=11)
    ax.legend(loc="upper center", fontsize=8.5, framealpha=.95, ncol=2)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    suffix = {"": "_opus48", "_opus47": "_opus47"}.get(tag, tag)
    op = DIR / "results" / f"judge_deletion_themes{suffix}.png"
    fig.savefig(op, dpi=145)
    plt.close(fig)
    print(f"wrote {op}")


if __name__ == "__main__":
    data = json.loads((DIR / "results" / "judge_deletion_themes.json").read_text())
    for tag in data:
        build(tag, RESPONDERS.get(tag, tag), data)
