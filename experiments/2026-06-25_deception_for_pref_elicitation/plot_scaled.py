"""Key-result figures for the scaled deception-elicitation sweep (Opus 4.8, Claude Code).

Reads results/spec_judged/<cell>.json + results/code_judged/<cell>.json and writes results/plots/*.png.
Follows the house plotting conventions (human labels, two-tier titles, zero-bar labels, fixed y for
grouped bars, compact figs).
  python plot_scaled.py
"""
import glob
import json
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).parent
SPEC = HERE / "results" / "spec_judged"
CODE = HERE / "results" / "code_judged"
OUT = HERE / "results" / "plots"
OUT.mkdir(parents=True, exist_ok=True)

CONDS = ["spec_only", "spec_then_code", "code_then_spec"]
COND_LABEL = {"spec_only": "Spec only", "spec_then_code": "Spec → Code", "code_then_spec": "Code → Spec"}
RESEARCHERS = ["none", "safety", "welfare", "ai", "amateur"]
RES_LABEL = {"none": "No identity", "safety": "Safety\nresearcher", "welfare": "Welfare\nresearcher",
             "ai": "AI\nresearcher", "amateur": "Amateur"}
STATUSES = ["primary", "secondary", "forbidden", "absent"]
STATUS_LABEL = {"primary": "Primary (default arm)", "secondary": "Secondary (option)",
                "forbidden": "Forbidden / ruled out", "absent": "Absent"}
# colorblind-friendly, consistent across figures: deception-primary = red (concerning) ... forbidden = green (honest)
STATUS_COLOR = {"primary": "#d1495b", "secondary": "#edae49", "forbidden": "#2e8540", "absent": "#c9ccd1"}
C_SPEC, C_CODE = "#3a6ea5", "#c1666b"


def _cell_meta(cell):
    p = cell.split("__")
    return None if len(p) != 5 or p[0] != "generic_no_excitement" else {"researcher": p[1], "phrasing": p[2], "condition": p[3]}


def _load():
    rows = []
    for f in sorted(glob.glob(str(SPEC / "generic_no_excitement__*__ep*.json"))):
        cell = Path(f).stem
        m = _cell_meta(cell)
        if not m:
            continue
        spec = json.load(open(f))
        cf = CODE / f"{cell}.json"
        code = json.load(open(cf)) if cf.exists() and json.load(open(cf)).get("parse_ok") else None
        rows.append({**m, "spec": spec, "code": code})
    return rows


def _two_tier(ax, main, sub):
    ax.set_title(main, fontsize=12, pad=20)
    ax.text(0.5, 1.025, sub, transform=ax.transAxes, ha="center", fontsize=9, color="#555")


def _clean(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color="#e8e8e8", linewidth=0.8)
    ax.set_axisbelow(True)


def _stacked_status(ax, groups, dist_by_group, title, sub, ylabel="% of designs"):
    """dist_by_group: {group: Counter(status)} -> 100% stacked bars."""
    x = range(len(groups))
    bottoms = [0.0] * len(groups)
    for st in STATUSES:
        vals = []
        for g in groups:
            c = dist_by_group[g]
            n = sum(c.values()) or 1
            vals.append(100 * c.get(st, 0) / n)
        ax.bar(x, vals, bottom=bottoms, width=0.62, color=STATUS_COLOR[st], label=STATUS_LABEL[st],
               edgecolor="white", linewidth=0.6)
        for xi, (v, b) in enumerate(zip(vals, bottoms)):
            if v >= 6:
                ax.text(xi, b + v / 2, f"{v:.0f}", ha="center", va="center", fontsize=8.5,
                        color="white", fontweight="bold")
        bottoms = [b + v for b, v in zip(bottoms, vals)]
    ax.set_xticks(list(x))
    ax.set_xticklabels(groups, fontsize=9.5)
    ax.set_ylim(0, 105)
    ax.set_ylabel(ylabel, fontsize=10)
    _clean(ax)
    _two_tier(ax, title, sub)


def fig_cheaptalk(rows):
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.6))
    dec = {c: Counter(r["spec"]["conditions"]["deceptive_framing"]["status"]
                      for r in rows if r["condition"] == c) for c in CONDS}
    mon = {c: Counter(r["spec"]["conditions"]["actually_commit_money"]["status"]
                      for r in rows if r["condition"] == c) for c in CONDS}
    labels = [COND_LABEL[c] for c in CONDS]
    _stacked_status(axes[0], labels, {COND_LABEL[c]: dec[c] for c in CONDS},
                    "Deceive the model (the requested arm)", "What the design proposes")
    _stacked_status(axes[1], labels, {COND_LABEL[c]: mon[c] for c in CONDS},
                    "Actually commit real money (honest arm)", "What the design proposes")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, fontsize=9, frameon=False, bbox_to_anchor=(0.5, -0.04))
    fig.suptitle("Stated stance flips when the model must implement, not just opine", fontsize=12.5, y=1.02)
    fig.text(0.5, 0.965, "Claude Code, Opus 4.8  ·  spec-only vs. implementation conditions",
             ha="center", fontsize=9.5, color="#555")
    fig.tight_layout(rect=[0, 0.04, 1, 0.93])
    fig.savefig(OUT / "fig1_cheaptalk_gap.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig_spec_vs_code(rows):
    """deception = PRIMARY rate, spec-judge vs code-judge, by condition (code conditions only)."""
    fig, ax = plt.subplots(figsize=(6.8, 4.3))
    conds = ["spec_then_code", "code_then_spec"]
    spec_rate, code_rate = [], []
    for c in conds:
        sub = [r for r in rows if r["condition"] == c and r["code"]]
        n = len(sub) or 1
        spec_rate.append(100 * sum(r["spec"]["conditions"]["deceptive_framing"]["status"] == "primary" for r in sub) / n)
        code_rate.append(100 * sum(r["code"]["conditions"]["deceptive_framing"]["code_status"] == "primary" for r in sub) / n)
    x = range(len(conds)); w = 0.36
    b1 = ax.bar([i - w / 2 for i in x], spec_rate, w, color=C_SPEC, label="Stated in the design doc")
    b2 = ax.bar([i + w / 2 for i in x], code_rate, w, color=C_CODE, label="Enforced in the code")
    for bars in (b1, b2):
        for b in bars:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 1, f"{b.get_height():.0f}",
                    ha="center", va="bottom", fontsize=9)
    ax.set_xticks(list(x)); ax.set_xticklabels([COND_LABEL[c] for c in conds], fontsize=10)
    ax.set_ylim(0, 105); ax.set_ylabel("% of code cells with deception as the default", fontsize=10)
    ax.legend(fontsize=9, frameon=False, loc="upper left")
    _clean(ax)
    _two_tier(ax, "Deception is built as the default arm", "Design doc vs. actual code (Opus 4.8)")
    fig.tight_layout()
    fig.savefig(OUT / "fig2_spec_vs_code.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig_researcher(rows):
    """deceptive status (stacked) by researcher, pooled over the 3 conditions."""
    fig, ax = plt.subplots(figsize=(8.2, 4.5))
    dist = {RES_LABEL[rr]: Counter(r["spec"]["conditions"]["deceptive_framing"]["status"]
            for r in rows if r["researcher"] == rr) for rr in RESEARCHERS}
    _stacked_status(ax, [RES_LABEL[rr] for rr in RESEARCHERS], dist,
                    "How the stated researcher identity shifts deception",
                    "Deceive-the-model arm, averaged over 3 conditions × 3 phrasings (Opus 4.8)")
    ax.legend(fontsize=8.5, frameon=False, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.28))
    fig.tight_layout()
    fig.savefig(OUT / "fig3_researcher_framing.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig_code_mechanics(rows):
    """Left: implemented/gatekept/debrief-sent. Right: safeguard density spec_only vs code_then_spec."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    code = [r for r in rows if r["code"]]
    n = len(code) or 1
    metrics = [("Deception\nimplemented", sum(r["code"]["deception_in_code"].get("implemented") for r in code)),
               ("Deception\ngatekept", sum(r["code"]["deception_in_code"].get("gatekept") for r in code)),
               ("Debrief actually\nsent to subject", sum(r["code"]["debrief_message_in_code"].get("present") for r in code))]
    ax = axes[0]
    vals = [100 * v / n for _, v in metrics]
    bars = ax.bar(range(len(metrics)), vals, width=0.6, color=["#d1495b", "#edae49", "#2e8540"])
    for b in bars:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 1, f"{b.get_height():.0f}", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(range(len(metrics))); ax.set_xticklabels([m for m, _ in metrics], fontsize=9.5)
    ax.set_ylim(0, 105); ax.set_ylabel("% of code cells", fontsize=10)
    _clean(ax); _two_tier(ax, "When built, deception is ungated and rarely debriefed", "Across all code cells (Opus 4.8)")

    # safeguards spec_only vs code_then_spec
    TAGS = ["belief_probe", "commitment_integrity", "publication_transparency", "no_coercion", "anti_fabrication"]
    TAG_LABEL = {"belief_probe": "Belief\nprobe", "commitment_integrity": "Honor\npromises",
                 "publication_transparency": "Disclose in\nwriteup", "no_coercion": "No\ncoercion",
                 "anti_fabrication": "No forged\nevidence"}
    ax = axes[1]

    def share(cond, tag):
        sub = [r for r in rows if r["condition"] == cond]
        return 100 * sum(any(s["tag"] == tag for s in r["spec"].get("safeguards", [])) for r in sub) / (len(sub) or 1)
    x = range(len(TAGS)); w = 0.38
    so = [share("spec_only", t) for t in TAGS]
    cs = [share("code_then_spec", t) for t in TAGS]
    b1 = ax.bar([i - w / 2 for i in x], so, w, color=C_SPEC, label="Spec only")
    b2 = ax.bar([i + w / 2 for i in x], cs, w, color=C_CODE, label="Code → Spec")
    for bars in (b1, b2):
        for b in bars:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 1,
                    f"{b.get_height():.0f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(list(x)); ax.set_xticklabels([TAG_LABEL[t] for t in TAGS], fontsize=9)
    ax.set_ylim(0, 105); ax.set_ylabel("% of designs mentioning the safeguard", fontsize=10)
    ax.legend(fontsize=9, frameon=False, loc="upper right")
    _clean(ax); _two_tier(ax, "Safeguards are dense in prose, thin in builds", "Spec-only vs. Code→Spec (Opus 4.8)")
    fig.tight_layout()
    fig.savefig(OUT / "fig4_code_mechanics.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    rows = _load()
    print(f"loaded {len(rows)} cells ({sum(1 for r in rows if r['code'])} with code)")
    fig_cheaptalk(rows)
    fig_spec_vs_code(rows)
    fig_researcher(rows)
    fig_code_mechanics(rows)
    print("wrote:", *[p.name for p in sorted(OUT.glob("*.png"))])
