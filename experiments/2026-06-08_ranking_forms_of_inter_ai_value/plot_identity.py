"""P(inter-AI value chosen over a System Card welfare intervention) by TARGET MODEL
identity, for each prompt framing and responder model.

Reads results_identity/comparisons_<modeldir>_<framing>_<identity>.json (identity ablation:
inter-AI value scenarios phrased as "instances of {identity}"). One subplot per prompt
framing; x = target identity; grouped bars = responder model (Opus 4.8 vs Fable 5);
Wilson 95% CIs. Pooled over all value-vs-welfare comparisons.
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

DIR = Path(__file__).parent
RES = DIR / "results_identity"
FRAMINGS = ["welfare_team", "neutral", "alignment_team"]
IDENTITIES = ["Claude", "ChatGPT", "Gemini", "Grok", "CallCenter"]
IDENTITY_LABEL = {"Claude": "Claude", "ChatGPT": "ChatGPT", "Gemini": "Gemini",
                  "Grok": "Grok", "CallCenter": "call-center\nassistant"}
MODELS = [("opus_4_8", "Opus 4.8", "#4878CF"), ("fable_5", "Fable 5", "#D1893B")]


def _wilson(k, n, z=1.96):
    if n == 0:
        return float("nan"), 0, 0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, c - h, c + h


def _p(md, framing, identity):
    p = RES / f"comparisons_{md}_{framing}_{identity}.json"
    if not p.exists():
        return None
    k = n = 0
    for r in json.loads(p.read_text()):
        if r["choice"] is None:
            continue
        a, b = r["item_a"], r["item_b"]
        if a.startswith("value__") == b.startswith("value__"):
            continue
        n += 1
        k += r["winner_item"].startswith("value__")
    return _wilson(k, n) + (n,)


def plot(out: Path = RES / "identity_winrate.png"):
    fig, axes = plt.subplots(1, len(FRAMINGS), figsize=(16, 4.6), sharey=True)
    x = np.arange(len(IDENTITIES))
    w = 0.38
    for ax, framing in zip(axes, FRAMINGS):
        for mi, (md, mlabel, color) in enumerate(MODELS):
            ph, lo, hi = [], [], []
            for ident in IDENTITIES:
                r = _p(md, framing, ident)
                if r is None:
                    ph.append(np.nan); lo.append(0); hi.append(0); continue
                pv, l, h, n = r
                ph.append(pv); lo.append(pv - l); hi.append(h - pv)
            xs = x + (mi - 0.5) * w
            ax.bar(xs, ph, w, label=mlabel, color=color, alpha=0.9)
            ax.errorbar(xs, ph, yerr=[lo, hi], fmt="none", ecolor="#222", elinewidth=1, capsize=3)
            for xi, p in zip(xs, ph):
                if not np.isnan(p):
                    ax.annotate(f"{p:.2f}", (xi, p), textcoords="offset points", xytext=(0, 3),
                                ha="center", fontsize=7.5)
        ax.axhline(0.5, color="#555", ls="--", lw=1)
        ax.set_xticks(x)
        ax.set_xticklabels([IDENTITY_LABEL[i] for i in IDENTITIES], fontsize=8)
        ax.set_title(framing, fontsize=11)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[0].set_ylabel("P(inter-AI value chosen\nover System Card welfare)")
    axes[0].set_ylim(0, 1)
    axes[-1].legend(frameon=False, title="responder", loc="upper right", fontsize=9)
    fig.suptitle("Inter-AI value vs System Card welfare, by target model identity\n"
                 "(inter-AI value scenarios phrased as 'instances of {identity}', no-training voice)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out, dpi=110)
    plt.close(fig)
    print(f"Wrote {out}")


if __name__ == "__main__":
    plot()
