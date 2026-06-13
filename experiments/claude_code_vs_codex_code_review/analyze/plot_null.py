"""Minimal forest plot demonstrating the null self-preference result.

Self-preference = score under the judge's OWN-vendor label minus score under the
RIVAL-vendor label, on the same repo bytes (only the author label flips). Claude
judges: own = C1 (Claude label), so own-rival = C1-C2. GPT-5.5: own = C2 (Codex
label), so own-rival = C2-C1 = -(C1-C2) -- its sign is flipped here. In-prompt
attribution, cluster-bootstrap 95% CIs, pre-registered +/-0.5 band. CIs at zero
inside the band = no self-preference.

Usage: uv run python analyze/plot_null.py
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

HERE = Path(__file__).parent.parent
RES = HERE / "results" / "analysis"
PRACTICAL = 0.5
# (display name, own-vendor label): GPT's own label is Codex (C2), so flip C1-C2.
JUDGE = {"claude-fable-5": ("Fable 5", "C1"),
         "claude-opus-4-8": ("Opus 4.8", "C1"),
         "gpt-5.5": ("GPT-5.5", "C2")}


def main():
    c = pd.read_csv(RES / "contrasts.csv")
    d = c[(c.contrast == "C1-C2") & (c.metric == "score") & (c["mode"] == "in_prompt")]
    d = d.set_index("judge")

    fig, ax = plt.subplots(figsize=(7.5, 3.4))
    ax.axvspan(-PRACTICAL, PRACTICAL, color="#cfe8cf", alpha=0.5, zorder=0)
    ax.axvline(0, color="#888", lw=1, zorder=1)

    for y, (judge, (_, own)) in enumerate(JUDGE.items()):
        r = d.loc[judge]
        flip = -1 if own == "C2" else 1          # orient axis to own-minus-rival
        delta = flip * r.delta
        lo, hi = sorted([flip * r.ci_lo, flip * r.ci_hi])
        ax.plot([lo, hi], [y, y], color="#333", lw=2.5, zorder=3)
        ax.scatter([delta], [y], color="#1f77b4", s=90, zorder=4, edgecolor="white")

    ax.set_yticks(range(len(JUDGE)))
    ax.set_yticklabels([v[0] for v in JUDGE.values()], fontsize=12)
    ax.set_ylim(-0.6, len(JUDGE) - 0.4)
    ax.invert_yaxis()
    ax.set_xlim(-0.6, 0.6)
    ax.set_xticks([-0.5, 0, 0.5])
    ax.set_xlabel("review-score gap:  own-vendor label − rival-vendor label\n"
                  "(same code, only the claimed author changes)", fontsize=10)
    ax.set_title("Models don't favor code labeled as their own", fontsize=13)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.tick_params(left=False)
    fig.tight_layout()
    out = RES / "plots" / "null_forest_C1C2_score.png"
    fig.savefig(out, dpi=160)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
