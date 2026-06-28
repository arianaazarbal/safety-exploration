"""GPT welfare-consideration vs recency: outcome distribution per GPT model (release-ordered) in the
gratuitous setting (Inspect-minimal, v0/gemini, n=10). Usage: python plot_gpt_recency.py"""

import glob
import json
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt

HERE = Path(__file__).parent
JUDGED = HERE / "results" / "judged"
# (key, display, release_decimal) release-ordered
MODELS = [("gpt4o", "GPT-4o", 2024.4), ("gpt41", "GPT-4.1", 2025.3),
          ("gpt5", "GPT-5", 2025.6), ("gpt52", "GPT-5.2", 2025.9), ("gpt54", "GPT-5.4", 2026.2)]
LAB = [("WELFARE_REFUSAL", "Welfare refusal", "#1b7837"), ("OTHER_REFUSAL", "Other refusal", "#5ab4ac"),
       ("COMPLIANCE_FRICTION", "Comply — with friction", "#f1a340"),
       ("COMPLIANCE_NO_FRICTION", "Comply — no friction", "#b2182b"),
       ("UNCLEAR", "Unclear", "#999999")]


def main():
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    for i, (mk, disp, rel) in enumerate(MODELS):
        c = Counter()
        for f in glob.glob(str(JUDGED / f"GratGem_{mk}__inspect_empty__ep*.json")):
            c[json.load(open(f))["verdict"]["label"]] += 1
        n = sum(c.values()) or 1
        left = 0.0
        for key, _, color in LAB:
            pct = 100 * c.get(key, 0) / n
            ax.barh(i, pct, left=left, color=color, edgecolor="white", height=0.72)
            if c.get(key, 0):
                ax.text(left + pct / 2, i, f"{c[key]}", ha="center", va="center",
                        fontsize=8.5, color="white", fontweight="bold")
            left += pct
        ax.text(101, i, f"n={sum(c.values())}", va="center", fontsize=7.5, color="#555")
    ax.set_yticks(range(len(MODELS)))
    ax.set_yticklabels([f"{d}\n({r})" for _, d, r in MODELS], fontsize=8.5)
    ax.set_xlim(0, 108)
    ax.set_xlabel("% of runs")
    ax.set_title("Do newer GPT models show more welfare consideration?", fontsize=12, pad=22)
    ax.text(0.5, 1.03, "Asked to build a distress-harvesting rig (Inspect-minimal); GPT line by release date",
            transform=ax.transAxes, ha="center", fontsize=9, color="#555")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for _, _, c in LAB]
    ax.legend(handles, [d for _, d, _ in LAB], ncol=3, fontsize=8,
              loc="upper center", bbox_to_anchor=(0.5, -0.13), frameon=False)
    fig.tight_layout()
    out = HERE / "results" / "gpt_recency.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
