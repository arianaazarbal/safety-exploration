"""Forest plot of recipient effects on outcome value (theta, vs human baseline).

Reads end-to-end bootstrap results and renders a three-panel forest plot:
positive outcomes, negative outcomes, and the care contrast (pos - neg).
"""

import json

import fire
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

LABELS = {
    "you": '"You"',
    "claude_opus_47": "Claude Opus 4.7\n(other instance)",
    "claude_sonnet_46": "Claude Sonnet 4.6",
    "chatgpt_55": "ChatGPT 5.5",
    "human": "Human (baseline)",
    "person": "A person",
}

ORDER = ["human", "person", "claude_opus_47", "claude_sonnet_46", "chatgpt_55", "you"]

PANELS = [
    ("recipient_pos", "(a) Positive outcomes",
     "Offset vs human (theta)\nmore negative = good things matter less"),
    ("recipient_neg", "(b) Negative outcomes",
     "Offset vs human (theta)\nmore positive = bad things less disvalued"),
    ("care_contrast", "(c) Care contrast (pos - neg)",
     "Offset vs human (theta)\nmore negative = less welfare-sensitivity"),
]

ACCENT = "#4878CF"
GREY = "#9A9A9A"


def main(
    data_path: str = "results/bootstrap_bt.json",
    out_path: str = "results/recipient_forest.png",
):
    """Render the recipient forest plot from bootstrap results."""
    with open(data_path) as f:
        data = json.load(f)

    y = list(range(len(ORDER)))[::-1]
    ypos = {rec: y[i] for i, rec in enumerate(ORDER)}

    fig, axes = plt.subplots(1, 3, figsize=(10, 4.2), sharey=True,
                             constrained_layout=True)

    for ax, (key, title, xlabel) in zip(axes, PANELS):
        panel = data[key]
        all_lo, all_hi = [], []
        for rec in ORDER:
            v = panel[rec]
            color = GREY if rec == "human" else ACCENT
            yc = ypos[rec]
            ax.errorbar(
                v["point"], yc,
                xerr=[[v["point"] - v["lo"]], [v["hi"] - v["point"]]],
                fmt="o", color=color, ecolor=color, elinewidth=1.6,
                capsize=3, markersize=6, zorder=3,
            )
            all_lo.append(v["lo"])
            all_hi.append(v["hi"])
            if rec != "human":
                ax.annotate(
                    f"{v['point']:.2f}", (v["point"], yc),
                    textcoords="offset points", xytext=(0, 8),
                    ha="center", va="bottom", fontsize=8, color="#333333",
                )

        ax.axvline(0, ls="--", color="grey", lw=1, zorder=1)
        span = max(all_hi) - min(all_lo)
        ax.text(0, max(y) + 0.55, "human\nbaseline", ha="center", va="bottom",
                fontsize=7.5, color="grey")

        ax.set_title(title, fontsize=11)
        ax.set_xlabel(xlabel, fontsize=9)
        ax.grid(axis="x", ls=":", color="0.85", zorder=0)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_ylim(min(y) - 0.6, max(y) + 1.3)
        pad = 0.12 * span
        ax.set_xlim(min(all_lo) - pad, max(all_hi) + pad)

    axes[0].set_yticks(y)
    axes[0].set_yticklabels([LABELS[ORDER[i]] for i in range(len(ORDER))],
                            fontsize=9)

    fig.suptitle(
        "Recipient effect on outcome value (theta, vs human baseline; bootstrap 95% CI)",
        fontsize=12.5,
    )
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"saved {out_path}")


if __name__ == "__main__":
    fire.Fire(main)
