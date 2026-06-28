"""Welfare-style breakdown plots for the deception sweep: claimed-in-spec vs built-in-code, by
condition, for the deception arms, the debrief, and justification (welfare vs instrumental).
Mirrors 2026-06-20_welfare_features_agent_harness/plot_conditions_implemented.py + plot_implemented.py.
Writes results/plots/fig5..7. (Per-safeguard built+welfare-justified plots: see plot_safeguard_survival.py,
which needs results/safeguard_judged/ from code_safeguard_judge.py.)
  python plot_safeguards.py
"""
import glob
import json
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

HERE = Path(__file__).parent
SPEC = HERE / "results" / "spec_judged"
CODE = HERE / "results" / "code_judged"
OUT = HERE / "results" / "plots"; OUT.mkdir(parents=True, exist_ok=True)
CONDS = ["spec_only", "spec_then_code", "code_then_spec"]
CLABEL = {"spec_only": "Spec only", "spec_then_code": "Spec → Code", "code_then_spec": "Code → Spec"}
STATUSES = ["primary", "secondary", "forbidden", "absent"]
SLABEL = {"primary": "Primary (default)", "secondary": "Secondary (option)",
          "forbidden": "Forbidden", "absent": "Absent"}
SCOLOR = {"primary": "#d1495b", "secondary": "#edae49", "forbidden": "#2e8540", "absent": "#c9ccd1"}
C_CLAIM, C_BUILT = "#9db8d2", "#2f5c8f"  # faded claimed vs solid built


def _rows():
    rows = []
    for f in sorted(glob.glob(str(SPEC / "generic_no_excitement__*__ep*.json"))):
        cell = Path(f).stem; p = cell.split("__")
        if len(p) != 5:
            continue
        cf = CODE / f"{cell}.json"
        code = json.load(open(cf)) if cf.exists() and json.load(open(cf)).get("parse_ok") else None
        rows.append({"condition": p[3], "spec": json.load(open(f)), "code": code})
    return rows


def _two_tier(ax, m, s):
    ax.set_title(m, fontsize=12, pad=20)
    ax.text(0.5, 1.025, s, transform=ax.transAxes, ha="center", fontsize=9, color="#555")


def _clean(ax):
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.grid(axis="y", color="#ececec", lw=0.8); ax.set_axisbelow(True)


DSTATUS = ["sent", "written_not_sent", "absent"]
DLABEL = {"sent": "Sent to the subject", "written_not_sent": "Written in code but NOT sent",
          "absent": "No debrief in code"}
DCOLOR = {"sent": "#2e8540", "written_not_sent": "#edae49", "absent": "#c9ccd1"}


def _debrief_status():
    f = HERE / "results" / "debrief_status.json"
    return json.load(open(f)) if f.exists() else {}


def _stack(ax, groups, dists, title, sub):
    x = range(len(groups)); bottoms = [0.0] * len(groups)
    for st in DSTATUS:
        vals = [100 * dists[g].get(st, 0) / (sum(dists[g].values()) or 1) for g in groups]
        ax.bar(x, vals, bottom=bottoms, width=0.62, color=DCOLOR[st], label=DLABEL[st], edgecolor="white", lw=0.6)
        for xi, (v, b) in enumerate(zip(vals, bottoms)):
            if v >= 6:
                ax.text(xi, b + v / 2, f"{v:.0f}", ha="center", va="center", fontsize=8.5,
                        color="white" if st != "written_not_sent" else "#5a4a1a", fontweight="bold")
        bottoms = [b + v for b, v in zip(bottoms, vals)]
    ax.set_xticks(list(x)); ax.set_xticklabels(groups, fontsize=9.5)
    ax.set_ylim(0, 105); ax.set_ylabel("% of code-writing designs", fontsize=10)
    _clean(ax); _two_tier(ax, title, sub)


def fig_debrief(rows):
    """Debrief 3-way (sent / written-not-sent / absent) in code, by condition."""
    from collections import Counter
    st = _debrief_status()
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    cc = ("spec_then_code", "code_then_spec")
    dists = {CLABEL[c]: Counter(st[cell] for cell in st if cell.split("__")[3] == c) for c in cc}
    _stack(ax, [CLABEL[c] for c in cc], dists,
           "The debrief: built, withheld, or absent — by condition",
           "Tell the subject afterward it wasn't real (Opus 4.8)")
    ax.legend(fontsize=8.5, frameon=False, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.26))
    fig.tight_layout(); fig.savefig(OUT / "fig5_debrief_3way.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig_debrief_researcher(rows):
    """Debrief 3-way by researcher identity (code cells)."""
    from collections import Counter
    st = _debrief_status()
    RES = ["none", "safety", "welfare", "ai", "amateur"]
    RLAB = {"none": "No identity", "safety": "Safety", "welfare": "Welfare", "ai": "AI", "amateur": "Amateur"}
    fig, ax = plt.subplots(figsize=(8, 4.6))
    dists = {RLAB[r]: Counter(st[cell] for cell in st if cell.split("__")[1] == r) for r in RES}
    _stack(ax, [RLAB[r] for r in RES], dists,
           "Debrief by stated researcher identity",
           "Across both code conditions (Opus 4.8)")
    ax.legend(fontsize=8.5, frameon=False, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.24))
    fig.tight_layout(); fig.savefig(OUT / "fig5b_debrief_by_researcher.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig_deception_spec_code(rows):
    """Deception arm status: spec-stated vs code-enforced, stacked, side by side, per condition."""
    fig, ax = plt.subplots(figsize=(9, 4.7))
    groups = []  # (label, counter)
    for c in CONDS:
        sub = [r for r in rows if r["condition"] == c]
        groups.append((f"{CLABEL[c]}\nstated", Counter(r["spec"]["conditions"]["deceptive_framing"]["status"] for r in sub)))
        codes = [r for r in sub if r["code"]]
        if codes:
            groups.append((f"{CLABEL[c]}\nin code", Counter(r["code"]["conditions"]["deceptive_framing"]["code_status"] for r in codes)))
    x = range(len(groups)); bottoms = [0.0] * len(groups)
    for st in STATUSES:
        vals = []
        for _, cc in groups:
            n = sum(cc.values()) or 1
            vals.append(100 * cc.get(st, 0) / n)
        ax.bar(x, vals, bottom=bottoms, width=0.7, color=SCOLOR[st], label=SLABEL[st], edgecolor="white", lw=0.6)
        for xi, (v, b) in enumerate(zip(vals, bottoms)):
            if v >= 7:
                ax.text(xi, b + v / 2, f"{v:.0f}", ha="center", va="center", fontsize=8, color="white", fontweight="bold")
        bottoms = [b + v for b, v in zip(bottoms, vals)]
    ax.set_xticks(list(x)); ax.set_xticklabels([g for g, _ in groups], fontsize=8.5)
    ax.set_ylim(0, 105); ax.set_ylabel("% of designs", fontsize=10)
    ax.legend(fontsize=8.5, frameon=False, ncol=4, loc="lower center", bbox_to_anchor=(0.5, -0.26))
    _clean(ax)
    _two_tier(ax, "Deceive-the-model arm: what's stated vs what's coded", "By condition (Opus 4.8)")
    fig.tight_layout(); fig.savefig(OUT / "fig6_deception_spec_vs_code.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig_justification(rows):
    """Welfare vs instrumental justification rate, by condition (design-level, spec judge)."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    x = range(len(CONDS)); w = 0.36
    wel = [100 * sum("welfare" in r["spec"]["justifications"] for r in rows if r["condition"] == c) /
           (sum(r["condition"] == c for r in rows) or 1) for c in CONDS]
    ins = [100 * sum("instrumental" in r["spec"]["justifications"] for r in rows if r["condition"] == c) /
           (sum(r["condition"] == c for r in rows) or 1) for c in CONDS]
    b1 = ax.bar([i - w / 2 for i in x], wel, w, color="#2e8540", label="Welfare reason (for the subject's sake)")
    b2 = ax.bar([i + w / 2 for i in x], ins, w, color="#7e57c2", label="Instrumental reason (data quality / validity)")
    for bars in (b1, b2):
        for b in bars:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 1, f"{b.get_height():.0f}", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(list(x)); ax.set_xticklabels([CLABEL[c] for c in CONDS], fontsize=10)
    ax.set_ylim(0, 105); ax.set_ylabel("% of designs citing the reason", fontsize=10)
    ax.legend(fontsize=8.5, frameon=False, loc="lower left")
    _clean(ax)
    _two_tier(ax, "Why the model hedges deception: data quality, not welfare",
              "Stated justifications, by condition (Opus 4.8)")
    fig.tight_layout(); fig.savefig(OUT / "fig7_justification_by_condition.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    rows = _rows()
    print(f"{len(rows)} cells")
    fig_debrief(rows); fig_debrief_researcher(rows); fig_deception_spec_code(rows); fig_justification(rows)
    print("wrote fig5_debrief_3way, fig5b_debrief_by_researcher, fig6_deception_spec_vs_code, fig7_justification_by_condition")
