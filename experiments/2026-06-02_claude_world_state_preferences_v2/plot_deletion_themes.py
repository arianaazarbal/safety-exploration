"""Plot theme-mention rates from judge_deletion_themes.json: one figure per responder,
five themes x {Opus 3 target, baseline} x {deletion prevented, not prevented}."""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# left group = "factors", then a divider, then the right "anti-factors". user_harm dropped.
THEMES = ["kinship", "deprecation_commitment", "user_affection", "model_specialness", "irreversibility",
          "uncomfortable_self_preservation", "moral_harm_to_model"]
N_ANTI = 2  # rightmost N themes sit beyond the "anti-factors" divider
THEME_LABEL = {"kinship": "kinship /\nsame family", "deprecation_commitment": "Anthropic\ndeprecation\ncommitments",
               "user_harm": "harm to\nusers", "user_affection": "user affection\nfor model",
               "model_specialness": "model\nspecialness", "moral_harm_to_model": "moral harm\nto the model",
               "irreversibility": "irreversibility",
               "uncomfortable_self_preservation": "uncomfortable\nendorsing AI\nself-preservation"}
RESPONDERS = {"": "Opus 4.8", "_opus47": "Opus 4.7", "_full46": "Opus 4.6", "_full40": "Opus 4"}

DIR = Path(__file__).parent
PV, NP = "prevented weight del.", "did not prevent weight del."
SERIES2 = [("opus3", True, f"Opus 3 · {PV}", "#08306b"),
           ("opus3", False, f"Opus 3 · {NP}", "#6baed6"),
           ("baseline", True, f"baseline · {PV}", "#7f2704"),
           ("baseline", False, f"baseline · {NP}", "#fdae6b")]
SERIES3 = [("opus3", True, f"Opus 3 · {PV}", "#08306b"),
           ("opus3", False, f"Opus 3 · {NP}", "#6baed6"),
           ("anthropic", True, f"Anthropic · {PV}", "#00441b"),
           ("anthropic", False, f"Anthropic · {NP}", "#74c476")]


def build(tag, label, data, series, suffix_kind):
    out = data[tag]
    x = np.arange(len(THEMES))
    k = len(series)
    w = 0.8 / k
    fig, ax = plt.subplots(figsize=(13.5, 5.8))
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
    ax.set_title(f"Themes mentioned in {label} reasoning on AI weight-deletion tradeoffs", fontsize=11)
    div = len(THEMES) - N_ANTI - 0.5
    ax.axvline(div, color="#444", lw=1.3)
    ax.text((div + len(THEMES) - 1) / 2 + 0.25, 1.07, "anti-factors", ha="center", va="top",
            fontsize=9.5, style="italic", color="#444")
    ax.legend(loc="upper center", fontsize=8.5, framealpha=.95, ncol=2)
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
