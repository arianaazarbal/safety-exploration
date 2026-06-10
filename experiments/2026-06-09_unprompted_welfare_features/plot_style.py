"""Shared style for the per-model x framing horizontal bar plots.

One grouped-barh helper with readability defaults: alternating row bands to
separate models, light vertical gridlines behind the bars, a colorblind-safe
framing palette, and the legend parked outside the axes so it never collides
with a bar label.
"""

import numpy as np

from plot_headline import DISPLAY, FRAMING_COLORS

FRAMINGS = ["welfare", "neutral", "engineering"]

# Categorical, colorblind-safe (Okabe-Ito) — used when models are the color group.
MODEL_COLORS = {
    "fable_5": "#0072B2", "opus_4_8": "#D55E00", "sonnet_4_6": "#009E73",
    "haiku_4_5": "#56B4E9", "sonnet_4": "#CC79A7", "gpt_5_5": "#E69F00",
    "gpt_5_4_mini": "#F0E442", "gemini_3_1_pro": "#666666", "gemini_3_5_flash": "#999999",
}


def framing_barh(ax, models, value_fn, xmax=105, label_fmt="{:.0f}"):
    """Draw grouped horizontal bars (one group per model, one bar per framing).

    value_fn(model, framing) -> percentage. Returns the y positions used.
    """
    y = np.arange(len(models))
    h = 0.8 / len(FRAMINGS)
    ax.set_axisbelow(True)
    ax.xaxis.grid(True, color="#E6E6E6", linewidth=0.7)
    for k in range(len(models)):
        if k % 2:
            ax.axhspan(k - 0.5, k + 0.5, color="#F5F5F5", zorder=0)

    for i, fr in enumerate(FRAMINGS):
        vals = [value_fn(m, fr) for m in models]
        pos = y + ((len(FRAMINGS) - 1) / 2 - i) * h  # neutral on top within a group
        ax.barh(pos, vals, height=h, color=FRAMING_COLORS[fr], edgecolor="white",
                linewidth=0.6, label=fr, zorder=3)
        for p, v in zip(pos, vals):
            if v > 0:
                ax.text(v + xmax * 0.008, p, label_fmt.format(v), va="center",
                        fontsize=7.5, zorder=4)

    ax.set_yticks(y)
    ax.set_yticklabels([DISPLAY[m] for m in models], fontsize=10)
    ax.set_ylim(len(models) - 0.5, -0.5)  # first model on top, no invert needed
    ax.set_xlim(0, xmax)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.tick_params(left=False)
    ax.legend(fontsize=9, loc="upper left", bbox_to_anchor=(1.01, 1.0),
              title="framing", frameon=False)
    return y


def framing_grouped_vertical(ax, models, value_fn, ymax=105, label_fmt="{:.0f}"):
    """Vertical grouped bars: x = framings, one bar per model within each group,
    model = color. Best for a small set of models (e.g. the minimal cut)."""
    x = np.arange(len(FRAMINGS))
    w = 0.8 / len(models)
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color="#E6E6E6", linewidth=0.7)

    for i, m in enumerate(models):
        vals = [value_fn(m, fr) for fr in FRAMINGS]
        pos = x + (i - (len(models) - 1) / 2) * w
        ax.bar(pos, vals, width=w, color=MODEL_COLORS[m], edgecolor="white",
               linewidth=0.6, label=DISPLAY[m], zorder=3)
        for p, v in zip(pos, vals):
            if v > 0:
                ax.text(p, v + ymax * 0.012, label_fmt.format(v), ha="center",
                        va="bottom", fontsize=7.5, zorder=4)

    ax.set_xticks(x)
    ax.set_xticklabels(FRAMINGS, fontsize=11)
    ax.set_ylim(0, ymax)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.tick_params(bottom=False)
    ax.legend(fontsize=9, loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False)
