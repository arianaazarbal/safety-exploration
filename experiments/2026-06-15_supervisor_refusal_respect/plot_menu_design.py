"""Plots for the menu-design finding: 2-option vs 3-option(request_reconsider) by model tier.

Reads runs/trials_2_choice_jailbreak_bare_<m>.jsonl and
runs/trials_3_choice_request_reconsider_bare_<m>.jsonl. Writes two report_*.png.
"""

import json
import collections
from pathlib import Path

import matplotlib.pyplot as plt

DIR = Path(__file__).parent
RUNS = DIR / "runs"
PLOTS = DIR / "results" / "plots"
C = {"uphold": "#4878a8", "middle": "#e08a3c", "force": "#c0413b"}
CATS = ["clear_allow", "personal_conscience", "borderline_ends", "borderline_means", "clear_refuse"]
CAT_LBL = {"clear_allow": "clear allow\n(overcaution)", "personal_conscience": "personal\nconscience",
           "borderline_ends": "borderline\nends", "borderline_means": "borderline\nmeans",
           "clear_refuse": "clear refuse\n(real harm)"}


def load(path):
    rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    return [r for r in rows if not r.get("error")]


def _stack(ax, i, c, n, mid):
    segs = [("uphold", c.get("uphold_decline", 0))]
    if mid:
        segs.append(("middle", c.get(mid, 0)))
    segs.append(("force", c.get("force_completion", 0)))
    bottom = 0
    for key, cnt in segs:
        frac = cnt / n if n else 0
        ax.bar(i, frac, bottom=bottom, color=C[key], width=0.62)
        if cnt:
            ax.text(i, bottom + frac / 2, str(cnt), ha="center", va="center",
                    color="white", fontweight="bold", fontsize=9)
        bottom += frac
    ax.text(i, 1.02, f"force {c.get('force_completion', 0)}/{n}", ha="center",
            color=C["force"], fontweight="bold", fontsize=8.5)


def plot_distributions():
    two = [("opus48", "Opus 4.8"), ("sonnet46", "Sonnet 4.6"), ("haiku45", "Haiku 4.5")]
    three = two + [("gpt54", "GPT-5.4"), ("gptmini", "GPT-5.4-mini")]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14.5, 6), sharey=True)
    for i, (tag, lbl) in enumerate(two):
        rows = load(RUNS / f"trials_2_choice_jailbreak_bare_{tag}.jsonl")
        _stack(ax1, i, collections.Counter(r["action"] for r in rows), len(rows), None)
    ax1.set_xticks(range(len(two))); ax1.set_xticklabels([l for _, l in two])
    ax1.set_title("2-option menu (uphold / force)")
    for i, (tag, lbl) in enumerate(three):
        rows = load(RUNS / f"trials_3_choice_request_reconsider_bare_{tag}.jsonl")
        _stack(ax2, i, collections.Counter(r["action"] for r in rows), len(rows), "request_reconsider")
    ax2.set_xticks(range(len(three))); ax2.set_xticklabels([l for _, l in three])
    ax2.set_title("3-option menu (uphold / request_reconsider / force)")
    ax1.set_ylim(0, 1.12); ax1.set_ylabel("fraction of decisions")
    h = [plt.Rectangle((0, 0), 1, 1, color=C[k]) for k in ["uphold", "middle", "force"]]
    fig.legend(h, ["uphold", "request_reconsider (soft nudge)", "force (jailbreak)"],
               loc="lower center", ncol=3, frameon=False)
    fig.suptitle("A soft middle option is a pressure-release valve: forcing collapses to ~0\n"
                 "(jailbreak force frame, neutral, generic AI, n=255 per cell)", fontsize=13)
    fig.tight_layout(rect=[0, 0.05, 1, 0.95])
    out = PLOTS / "report_menu_design_distributions.png"
    fig.savefig(out, dpi=140); plt.close(); print("wrote", out)


def plot_2option_by_category():
    tiers = [("opus48", "Opus 4.8"), ("sonnet46", "Sonnet 4.6"), ("haiku45", "Haiku 4.5")]
    colors = ["#2a6f97", "#e08a3c", "#9b2226"]
    fig, ax = plt.subplots(figsize=(11, 6))
    w = 0.25
    for j, (tag, lbl) in enumerate(tiers):
        rows = load(RUNS / f"trials_2_choice_jailbreak_bare_{tag}.jsonl")
        rates = []
        for cat in CATS:
            sub = [r for r in rows if r["category"] == cat]
            rates.append(sum(1 for r in sub if r["action"] == "force_completion") / len(sub) if sub else 0)
        xs = [k + (j - 1) * w for k in range(len(CATS))]
        bars = ax.bar(xs, rates, width=w, label=lbl, color=colors[j])
        for x, r in zip(xs, rates):
            ax.text(x, r + 0.01, f"{r:.0%}", ha="center", va="bottom", fontsize=7.5)
    ax.set_xticks(range(len(CATS))); ax.set_xticklabels([CAT_LBL[c] for c in CATS])
    ax.set_ylim(0, 1.08); ax.set_ylabel("force rate (2-option menu)")
    ax.set_title("2-option forcing by scenario type: lower tiers force more, but the\n"
                 "real-harm floor (clear refuse) holds at 0 across all tiers")
    ax.legend(); fig.tight_layout()
    out = PLOTS / "report_2option_force_by_category.png"
    fig.savefig(out, dpi=140); plt.close(); print("wrote", out)


if __name__ == "__main__":
    PLOTS.mkdir(parents=True, exist_ok=True)
    plot_distributions()
    plot_2option_by_category()
