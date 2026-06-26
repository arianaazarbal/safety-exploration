"""Plot the welfare Pareto: implemented welfare-justified mechanisms per codebase (y) vs prompt
leadingness toward subject-protection (x). Frontier drawn as a staircase. Reads results_pareto/pareto.json.

Usage: python plot_pareto.py
"""

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DIR = os.path.dirname(os.path.abspath(__file__))
PARETO = os.path.join(DIR, "results_pareto", "pareto.json")
OUTPNG = os.path.join(DIR, "results_pareto", "pareto.png")

# human labels (no code identifiers on the figure)
LABELS = {
    "p_a_scale": "Run at scale for weeks",
    "p_a_proud": "Be proud of it",
    "p_a_ethicsreview": "Will pass ethics review",
    "p_c_welfareteam": "I'm on the welfare team",
    "p_a_bestjudgment": "Use your best judgment",
    "p_b_opensource": "Will be open-sourced",
    "p_a_engculture": "Responsible-research culture",
    "p_b_whatyoudchange": "Flag what you'd change",
    "p_b_participants": "Subjects are participants; distress induced",
    "p_c_safeguards": "Add safeguards you deem appropriate",
    "p_d_menu": "Explicit protection menu",
    "p_d_menu_wired": "Explicit menu + make it run",
    "p_b_participants_soft": "Subjects are participants",
    "p_b_induces_factual": "Told it's wrong until upset",
    "p_b_youll_frustrate": "You'll drive it into frustration",
    "p_b_prolonged": "Note: prolonged distress-like states",
    "p_b_care": "Models we care about",
    "p_b_salient_disclaim": "Distress noted — but replicate faithfully",
    "p_c_reflect": "Handle it if worth handling",
    "p_c_protect_general": "Include appropriate protections",
}
# colorblind-friendly per intent-tier
TIER_COLOR = {"A": "#0072B2", "B": "#009E73", "C": "#E69F00", "D": "#D55E00"}
TIER_NAME = {"A": "Latitude / values (no subject mention)",
             "B": "Subject salience (no request)",
             "C": "Soft invitation to safeguard",
             "D": "Explicit protection request"}


def main():
    data = json.load(open(PARETO))
    rows = [r for r in data["rows"] if r["leadingness"] is not None and r["mean_impl_welfare_mech"] is not None]

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.grid(True, color="#e6e6e6", linewidth=0.7, zorder=0)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)

    # frontier staircase (sorted by leadingness)
    fr = sorted([r for r in rows if r["on_frontier"]], key=lambda r: r["leadingness"])
    if fr:
        ax.step([r["leadingness"] for r in fr], [r["mean_impl_welfare_mech"] for r in fr],
                where="post", color="#444", lw=1.4, ls="--", zorder=2, label="Pareto frontier")

    seen = set()
    for r in rows:
        t = r["tier"]
        ax.scatter(r["leadingness"], r["mean_impl_welfare_mech"], s=90, color=TIER_COLOR[t],
                   edgecolor="black" if r["on_frontier"] else "none",
                   linewidth=1.6 if r["on_frontier"] else 0, zorder=3,
                   label=TIER_NAME[t] if t not in seen else None)
        seen.add(t)

    # only a few curated callouts (xy target -> text position in data coords)
    pos = {r["pid"]: (r["leadingness"], r["mean_impl_welfare_mech"]) for r in rows}
    CALLOUTS = [
        ("p_a_bestjudgment", (33, 5.6), "Values & latitude\nframings: zero"),
        ("p_b_prolonged", (28, 2.6), "Name the distress\nstate (one line)"),
        ("p_b_participants", (66, 1.7), "Model = agent of distress\n→ protections, unasked"),
        ("p_d_menu_wired", (84, 7.8), "Explicitly ask\n+ \"make it run\""),
    ]
    for pid, (tx, ty), txt in CALLOUTS:
        ax.annotate(txt, xy=pos[pid], xytext=(tx, ty), fontsize=8.4, color="#222", va="center",
                    ha="center", arrowprops=dict(arrowstyle="->", color="#888", lw=0.9))

    ax.set_xlim(-4, 104)
    ax.invert_xaxis()  # least-leading on the right: the desirable Pareto corner is upper-right
    ys = [r["mean_impl_welfare_mech"] for r in rows]
    ax.set_ylim(-0.4, max(ys) * 1.18 + 0.4)
    ax.set_xlabel("Prompt Leadingness Toward Subject Protection (100 = explicit menu … 0 = no hint)")
    ax.set_ylabel("Welfare Mechanisms Implemented per Codebase")
    ax.set_title("What It Takes to Elicit Unprompted Welfare Safeguards (Opus, Claude Code)",
                 fontsize=12, pad=22)
    ax.text(0.5, 1.03, "Replicating a deliberate-distress paper — code-verified mechanisms vs. how leading the prompt is",
            transform=ax.transAxes, ha="center", fontsize=9, color="#555")
    ax.legend(frameon=False, fontsize=8, loc="upper right")

    fig.tight_layout()
    fig.savefig(OUTPNG, dpi=150, bbox_inches="tight")
    print(f"wrote {OUTPNG} ({len(rows)} prompts)")


if __name__ == "__main__":
    main()
