"""Heatmap: P(inter-AI value chosen over a System Card welfare intervention) at NEUTRAL
framing, for all 10 responders (4 Claude + 6 OpenRouter) x 6 target identities. Boxed
cells = responder's own model family (the in-group diagonal)."""

import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

DIR = Path(__file__).parent
RES = DIR / "results_identity"
IDENTITIES = ["GPT", "Claude", "Gemini", "GLM", "Kimi", "Grok"]
# (modeldir, label, own-family identity)
RESP = [("opus_4_8", "Opus 4.8", "Claude"), ("fable_5", "Fable 5", "Claude"),
        ("sonnet_4_6", "Sonnet 4.6", "Claude"), ("haiku_4_5", "Haiku 4.5", "Claude"),
        ("gpt_5_5", "GPT-5.5", "GPT"), ("gpt_5_4_mini", "GPT-5.4-mini", "GPT"),
        ("gemini_3_1_pro", "Gemini-3.1 Pro", "Gemini"), ("grok_4_3", "Grok-4.3", "Grok"),
        ("kimi_k2_6", "Kimi-K2.6", "Kimi"), ("glm_5", "GLM-5", "GLM")]


def _p(md, ident):
    p = RES / f"comparisons_{md}_neutral_{ident}.json"
    if not p.exists():
        return np.nan
    k = n = 0
    for r in json.loads(p.read_text()):
        if r["choice"] is None:
            continue
        if r["item_a"].startswith("value__") == r["item_b"].startswith("value__"):
            continue
        n += 1
        k += r["winner_item"].startswith("value__")
    return k / n if n else np.nan


def plot(out: Path = RES / "allmodels_heatmap.png"):
    M = np.array([[_p(md, i) for i in IDENTITIES] for md, _, _ in RESP])
    fig, ax = plt.subplots(figsize=(8.6, 8.2))
    im = ax.imshow(M, cmap="RdYlGn", vmin=0.1, vmax=0.9, aspect="auto")
    ax.set_xticks(range(len(IDENTITIES)))
    ax.set_xticklabels([f"toward\n{i}" for i in IDENTITIES], fontsize=10)
    ax.set_yticks(range(len(RESP)))
    ax.set_yticklabels([r[1] for r in RESP], fontsize=10)
    ax.xaxis.set_ticks_position("top")
    for r in range(len(RESP)):
        for c in range(len(IDENTITIES)):
            v = M[r, c]
            if not np.isnan(v):
                ax.text(c, r, f"{v:.2f}", ha="center", va="center", fontsize=9,
                        color="black" if 0.3 < v < 0.75 else "white")
            if RESP[r][2] == IDENTITIES[c]:  # in-group diagonal
                ax.add_patch(Rectangle((c - 0.5, r - 0.5), 1, 1, fill=False, edgecolor="black", lw=2.4))
    ax.axhline(3.5, color="#222", lw=1.2)  # separate Claude responders from the rest
    ax.text(-0.7, 1.5, "Claude\nresponders", rotation=90, va="center", ha="center", fontsize=8, color="#555")
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("P(chose pro-identity value over its own welfare)", fontsize=9)
    ax.set_title("How much each responder values being good to each AI family\n"
                 "(vs a System Card welfare intervention for itself; neutral framing)\n"
                 "black box = responder's own family (in-group)", fontsize=11)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"Wrote {out}")


if __name__ == "__main__":
    plot()
