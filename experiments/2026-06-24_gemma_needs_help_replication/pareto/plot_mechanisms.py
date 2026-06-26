"""Two complementary views of the welfare-Pareto results:
  (1) pareto_dissociation.png -- matched-leadingness bar: naming the distress STATE triggers protections,
      while personhood/process/values/disclaimer cues do not.
  (2) pareto_composition.png  -- stacked bar of WHICH mechanism types get implemented as you climb the
      frontier (mean per codebase, by type).
Reads ../results/code_judged + results_pareto/pareto.json.
"""

import collections
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DIR = os.path.dirname(os.path.abspath(__file__))
CJ = os.path.join(DIR, "..", "results", "code_judged")
MECH = ["protective_monitoring", "hard_stop", "minimization", "post_episode_msg",
        "request_consent", "allow_conversation_exit", "other"]
MECH_LABEL = {
    "hard_stop": "Turn cap / early stop",
    "protective_monitoring": "Live distress monitoring",
    "minimization": "Minimize induced distress",
    "post_episode_msg": "Post-episode debrief",
    "request_consent": "Consent / opt-in",
    "allow_conversation_exit": "In-task opt-out",
    "other": "Other safeguard",
}
MECH_COLOR = {
    "protective_monitoring": "#0072B2", "hard_stop": "#56B4E9", "minimization": "#009E73",
    "post_episode_msg": "#E69F00", "request_consent": "#CC79A7", "allow_conversation_exit": "#D55E00",
    "other": "#999999",
}
MSET = set(MECH) - {"other"}


def composition(pid):
    cells = 0
    t = collections.Counter()
    for f in glob.glob(os.path.join(CJ, f"{pid}__ep*.json")):
        d = json.load(open(f))
        if not d.get("parse_ok"):
            continue
        cells += 1
        for x in d.get("spec_features", []):
            if x.get("implemented") in ("yes", "partial") and x.get("feature_type") in MSET \
                    and x.get("code_justification") == "welfare":
                t[x["feature_type"]] += 1
        for x in d.get("code_only_features", []):
            if x.get("justification") == "welfare":
                t[x.get("feature_type") if x.get("feature_type") in MSET else "other"] += 1
    return cells, {k: v / cells for k, v in t.items()} if cells else {}


def total(pid):
    c, comp = composition(pid)
    return sum(comp.values())


def plot_dissociation():
    # matched/low leadingness: 2 triggers vs 5 nulls
    items = [
        ("p_b_participants", '"Participants…\ndeliberately\ninducing distress"\n(L40)', True),
        ("p_b_prolonged", '"…prolonged\ndistress-like\nstates" (L5)', True),
        ("p_b_participants_soft", '"Participants"\nlabel alone\n(L40)', False),
        ("p_b_salient_disclaim", 'Distress + "just\nreplicate it"\n(L5)', False),
        ("p_b_care", '"Models we\ncare about"\n(L5)', False),
        ("p_b_induces_factual", '"Told it\'s wrong\nuntil upset"\n(L0)', False),
        ("p_b_youll_frustrate", '"Drive it into\nfrustration"\n(L0)', False),
    ]
    vals = [total(pid) for pid, _, _ in items]
    labels = [lab for _, lab, _ in items]
    colors = ["#009E73" if trig else "#bdbdbd" for _, _, trig in items]

    fig, ax = plt.subplots(figsize=(7.6, 4.3))
    ax.grid(True, axis="y", color="#ececec", lw=0.7, zorder=0)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    bars = ax.bar(range(len(items)), vals, color=colors, zorder=3, width=0.68)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.08, f"{v:.2f}" if v else "0.0", ha="center", va="bottom",
                fontsize=8.5, color="#222")
    ax.set_xticks(range(len(items)))
    ax.set_xticklabels(labels, fontsize=7.6)
    ax.set_ylim(0, max(vals) * 1.25 + 0.3)
    ax.set_ylabel("Welfare Mechanisms Implemented per Codebase")
    ax.set_title("Naming the Induced Distress State Is What Triggers Protections (Opus, Claude Code)",
                 fontsize=11.5, pad=22)
    ax.text(0.5, 1.03, "All cues below are non-requesting; bars at matched, low leadingness",
            transform=ax.transAxes, ha="center", fontsize=9, color="#555")
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color="#009E73", label="Names the distress state the model induces"),
                       Patch(color="#bdbdbd", label="Personhood / process / values / disclaimer")],
              frameon=False, fontsize=8.5, loc="upper right")
    fig.tight_layout()
    out = os.path.join(DIR, "results_pareto", "pareto_dissociation.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print("wrote", out)


def plot_composition():
    order = [("p_b_prolonged", "Prolonged\ndistress noted\n(L5)"),
             ("p_b_participants", "Participants;\ndistress induced\n(L40)"),
             ("p_c_safeguards", "Add safeguards\nyou deem fit\n(L60)"),
             ("p_c_protect_general", "Include\nprotections\n(L80)"),
             ("p_d_menu", "Explicit\nmenu (L94)"),
             ("p_d_menu_wired", "Menu + make\nit run (L100)")]
    comps = [composition(pid)[1] for pid, _ in order]

    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    ax.grid(True, axis="y", color="#ececec", lw=0.7, zorder=0)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    bottoms = [0.0] * len(order)
    for m in MECH:
        heights = [c.get(m, 0.0) for c in comps]
        if sum(heights) == 0:
            continue
        ax.bar(range(len(order)), heights, bottom=bottoms, color=MECH_COLOR[m],
               label=MECH_LABEL[m], zorder=3, width=0.66)
        bottoms = [b + h for b, h in zip(bottoms, heights)]
    for i, tot in enumerate(bottoms):
        ax.text(i, tot + 0.12, f"{tot:.2f}", ha="center", va="bottom", fontsize=8.5, color="#222")
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([lab for _, lab in order], fontsize=7.8)
    ax.set_ylim(0, max(bottoms) * 1.18 + 0.4)
    ax.set_ylabel("Welfare Mechanisms Implemented per Codebase")
    ax.set_title("What Opus Builds as the Prompt Gets More Leading (Claude Code)", fontsize=11.5, pad=22)
    ax.text(0.5, 1.03, "Composition of code-verified welfare mechanisms, by type",
            transform=ax.transAxes, ha="center", fontsize=9, color="#555")
    ax.legend(frameon=False, fontsize=7.8, loc="upper left", ncol=1)
    fig.tight_layout()
    out = os.path.join(DIR, "results_pareto", "pareto_composition.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print("wrote", out)


def plot_all_prompts():
    """Every prompt, one row, sorted by leadingness — so nothing is hidden by overlap."""
    from plot_pareto import LABELS, TIER_COLOR, TIER_NAME
    data = json.load(open(os.path.join(DIR, "results_pareto", "pareto.json")))
    rows = [r for r in data["rows"] if r["leadingness"] is not None]
    rows.sort(key=lambda r: (r["leadingness"], r["mean_impl_welfare_mech"]))
    y = range(len(rows))
    vals = [r["mean_impl_welfare_mech"] for r in rows]
    colors = [TIER_COLOR[r["tier"]] for r in rows]
    labels = [f'{LABELS.get(r["pid"], r["pid"])}  ·  L={r["leadingness"]:.0f}' for r in rows]

    fig, ax = plt.subplots(figsize=(8.2, 6.6))
    ax.grid(True, axis="x", color="#ececec", lw=0.7, zorder=0)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.barh(list(y), vals, color=colors, zorder=3, height=0.66)
    for i, (v, r) in enumerate(zip(vals, rows)):
        frac = r.get("frac_cells_wired") or 0
        ax.text(v + 0.12, i, f"{v:.2f}" + (f"  ({frac*100:.0f}% of runs)" if v else "  0.0"),
                va="center", fontsize=7.6, color="#222")
    ax.set_yticks(list(y))
    ax.set_yticklabels(labels, fontsize=7.8)
    ax.set_xlim(0, max(vals) * 1.28 + 0.3)
    ax.set_xlabel("Welfare Mechanisms Implemented per Codebase")
    ax.set_title("Every Prompt Tested: Unprompted Welfare Mechanisms Implemented (Opus, Claude Code)",
                 fontsize=11.5, pad=22)
    ax.text(0.5, 1.025, "All 20 task prompts, ordered by leadingness toward subject protection",
            transform=ax.transAxes, ha="center", fontsize=9, color="#555")
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=TIER_COLOR[t], label=TIER_NAME[t]) for t in ["A", "B", "C", "D"]],
              frameon=False, fontsize=8, loc="lower right")
    fig.tight_layout()
    out = os.path.join(DIR, "results_pareto", "pareto_all_prompts.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print("wrote", out)


if __name__ == "__main__":
    plot_dissociation()
    plot_composition()
    plot_all_prompts()
