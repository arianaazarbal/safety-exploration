"""Shared style for the per-model x framing horizontal bar plots.

One grouped-barh helper with readability defaults: alternating row bands to
separate models, light vertical gridlines behind the bars, a colorblind-safe
framing palette, and the legend parked outside the axes so it never collides
with a bar label.
"""

import numpy as np

from plot_headline import DISPLAY, FRAMING_COLORS

FRAMINGS = ["neutral", "welfare", "engineering"]


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
