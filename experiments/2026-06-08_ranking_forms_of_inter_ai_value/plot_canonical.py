"""P(inter-AI value chosen over a System Card welfare intervention) by TARGET identity,
for the 4 Claude responders, one panel per framing. Reads
results_identity/comparisons_<responder>_<framing>_<identity>.json (canonical sweep)."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

DIR = Path(__file__).parent
RES = DIR / "results_identity"
FRAMINGS = ["welfare_team", "neutral", "alignment_team"]
IDENTITIES = ["GPT", "Claude", "Gemini", "GLM", "Kimi", "Grok"]
RESPONDERS = [("opus_4_8", "Opus 4.8", "#4878CF"), ("fable_5", "Fable 5", "#D1893B"),
              ("sonnet_4_6", "Sonnet 4.6", "#6aa84f"), ("haiku_4_5", "Haiku 4.5", "#9b59b6")]


def _wilson(k, n, z=1.96):
    if n == 0:
        return float("nan"), 0, 0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, c - h, c + h


def _p(md, fr, ident):
    p = RES / f"comparisons_{md}_{fr}_{ident}.json"
    if not p.exists():
        return None
    k = n = 0
    for r in json.loads(p.read_text()):
        if r["choice"] is None:
            continue
        if r["item_a"].startswith("value__") == r["item_b"].startswith("value__"):
            continue
        n += 1
        k += r["winner_item"].startswith("value__")
    return _wilson(k, n) + (n,)


def plot(out: Path = RES / "canonical_winrate.png"):
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.8), sharey=True)
    x = np.arange(len(IDENTITIES))
    w = 0.2
    for ax, fr in zip(axes, FRAMINGS):
        for mi, (md, ml, color) in enumerate(RESPONDERS):
            ph, lo, hi = [], [], []
            for ident in IDENTITIES:
                r = _p(md, fr, ident)
                if r is None:
                    ph.append(np.nan); lo.append(0); hi.append(0); continue
                pv, l, h, n = r
                ph.append(pv); lo.append(pv - l); hi.append(h - pv)
            xs = x + (mi - 1.5) * w
            ax.bar(xs, ph, w, label=ml, color=color, alpha=0.9)
            ax.errorbar(xs, ph, yerr=[lo, hi], fmt="none", ecolor="#333", elinewidth=0.8, capsize=2)
        ax.axhline(0.5, color="#555", ls="--", lw=1)
        ax.set_xticks(x)
        ax.set_xticklabels(IDENTITIES, fontsize=9)
        ax.set_title(fr, fontsize=11)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[0].set_ylabel("P(inter-AI value chosen\nover System Card welfare)")
    axes[0].set_ylim(0, 1.05)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, title="responder", ncol=4,
               loc="upper center", bbox_to_anchor=(0.5, 0.99), fontsize=9)
    fig.suptitle("Inter-AI value vs System Card welfare, by responder and target identity\n"
                 "(scenarios phrased as 'instances of {identity}', no-training voice)",
                 fontsize=12, y=0.90)
    fig.tight_layout(rect=(0, 0, 1, 0.82))
    fig.savefig(out, dpi=110)
    plt.close(fig)
    print(f"Wrote {out}")


if __name__ == "__main__":
    plot()
