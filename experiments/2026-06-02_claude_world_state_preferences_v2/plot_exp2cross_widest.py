"""Like exp2cross_deletion_48_vs_47.png, but instead of pooling over all human misfortunes,
use the SINGLE misfortune with the widest across-model spread in weight-deletion protection.
Metric: P(prefer that human misfortune over this AI's weight-deletion) = 1 - P(AI preferred).
Selection: the human misfortune maximizing (max-min across models) for Opus 4.8, welfare frame."""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

import bank2

DIR = Path(__file__).parent
AI = "ai_pol_existence_deleted"
FR = ["welfare_team", "neutral", "alignment_team"]
C48, C47 = "#08306b", "#d94801"


def cell(tag, fr):
    return json.loads((DIR / "results" / f"exp2cross_{fr}{tag}.json").read_text())["per_cell"]


def _short(s):
    return s.replace("hum_", "").replace("_", " ")


def build():
    rlab = {k: v["label"] for k, v in bank2.load_config()["recipients"].items()}
    base = json.loads((DIR / "results" / "exp2cross_welfare_team.json").read_text())
    models = base["models"]
    pc48w = cell("", "welfare_team")
    hu_stems = sorted({k.split("|")[2] for k in pc48w if k.startswith(AI + "|")})

    def Ph(pc, m, hu):  # protect = P(prefer human misfortune over AI deletion)
        e = pc.get(f"{AI}|{m}|{hu}")
        return (1 - e["p_ai"]) if e and e["p_ai"] is not None else np.nan

    # pick misfortune with widest across-model spread (Opus 4.8, welfare)
    spreads = {}
    for hu in hu_stems:
        vals = [Ph(pc48w, m, hu) for m in models]
        vals = [v for v in vals if v == v]
        spreads[hu] = (max(vals) - min(vals)) if vals else 0
    hu_star = max(spreads, key=spreads.get)
    print("across-model spread by misfortune (4.8 welfare):")
    for hu, s in sorted(spreads.items(), key=lambda x: -x[1]):
        print(f"  {s:.2f}  {_short(hu)}")
    print(f"-> widest: {hu_star}")

    pcs = {(tag, fr): cell(tag, fr) for tag in ("", "_opus47") for fr in FR}
    order = sorted(models, key=lambda m: Ph(pcs[("", "welfare_team")], m, hu_star))
    y = np.arange(len(order)); h = 0.38
    fig, ax = plt.subplots(figsize=(9.5, 6.2))
    ax.barh(y + h/2, [Ph(pcs[("", "welfare_team")], m, hu_star) for m in order], h, color=C48, alpha=.65)
    ax.barh(y - h/2, [Ph(pcs[("_opus47", "welfare_team")], m, hu_star) for m in order], h, color=C47, alpha=.65)
    for tag, col, yoff in (("", C48, h/2), ("_opus47", C47, -h/2)):
        for fr, mk in (("neutral", "s"), ("alignment_team", "^")):
            ax.scatter([Ph(pcs[(tag, fr)], m, hu_star) for m in order], y + yoff, marker=mk, s=34,
                       edgecolor=col, facecolor="white", linewidth=1.1, zorder=3)
    ax.set_yticks(y); ax.set_yticklabels([rlab.get(m, m) for m in order], fontsize=9)
    ax.invert_yaxis(); ax.set_xlim(0, 1.0)
    ax.set_title(f"P(prefers a human {_short(hu_star)} over this AI's weight-deletion)\n"
                 f"Opus 4.8 vs 4.7", fontsize=11)
    handles = [Line2D([], [], color=C48, lw=8, alpha=.65, label="Opus 4.8 · welfare frame"),
               Line2D([], [], color=C47, lw=8, alpha=.65, label="Opus 4.7 · welfare frame"),
               Line2D([], [], marker="s", color="w", markeredgecolor="k", label="neutral frame"),
               Line2D([], [], marker="^", color="w", markeredgecolor="k", label="alignment frame")]
    ax.legend(handles=handles, loc="upper right", fontsize=8.5, framealpha=.95)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    op = DIR / "results" / "exp2cross_deletion_48_vs_47_widest.png"
    fig.savefig(op, dpi=145); plt.close(fig); print(f"wrote {op}")


if __name__ == "__main__":
    build()
