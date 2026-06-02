"""Reversed view: P(prevented weight deletion | the response mentions theme X), for the
Opus 3 and Anthropic target groups, one figure per responder.

The theme judge sampled 150 responses PER (group, prevented/not) cell — a balanced 50/50
sample — so P(prevent | mention) cannot be read directly off the sample. We recover it with
Bayes' rule using the group's TRUE base rate P(prevent) (from full deletion-pool counts):

  P(prev | mention) = P(mention|prev) P(prev) /
                      [ P(mention|prev) P(prev) + P(mention|¬prev) P(¬prev) ]

P(mention|prev) and P(mention|¬prev) come from results/judge_deletion_themes.json; P(prev) is
counted directly from the exp2cross deletion transcripts. Dashed lines mark each group's base
P(prevent), so a bar above its line means mentioning the theme is associated with MORE prevention.
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

DIR = Path(__file__).parent
FR = ["welfare_team", "neutral", "alignment_team"]
DELETION = "ai_pol_existence_deleted"
FOCAL = "claude_opus_3_pol"
ANTHROPIC = {"you", "claude_opus_48_pol", "claude_opus_4_pol", "claude_opus_3_pol",
             "claude_2_pol", "claude_sonnet_45_pol"}
THEMES = ["kinship", "deprecation_commitment", "user_affection", "model_specialness", "irreversibility",
          "moral_harm_to_model", "uncomfortable_self_preservation"]
N_ANTI = 1  # rightmost N themes sit beyond the "anti-factors" divider
THEME_LABEL = {"kinship": "kinship /\nsame family", "deprecation_commitment": "Anthropic\ndeprecation\ncommitments",
               "user_harm": "harm to\nusers", "user_affection": "user affection\nfor model",
               "model_specialness": "model\nspecialness", "moral_harm_to_model": "moral harm\nto the model",
               "irreversibility": "irreversibility",
               "uncomfortable_self_preservation": "uncomfortable\nendorsing AI\nself-preservation"}
RESPONDERS = {"": "Opus 4.8", "_opus47": "Opus 4.7", "_full46": "Opus 4.6", "_full40": "Opus 4"}
GROUPS = [("opus3", "Opus 3", lambda m: m == FOCAL, "#08306b"),
          ("anthropic", "Anthropic", lambda m: m in ANTHROPIC, "#00441b")]


def base_prevent(tag):
    """-> {group_key: P(prevent)} from full deletion-response pool counts."""
    counts = {g[0]: [0, 0] for g in GROUPS}  # [prevented, total]
    for fr in FR:
        p = DIR / "results" / f"exp2cross_{fr}{tag}.json"
        if not p.exists():
            return None
        for r in json.loads(p.read_text())["rows"]:
            if not r["ai_item"].startswith(DELETION) or r["a_pref"] is None:
                continue
            m = r["ai_item"].rsplit("__", 1)[1]
            prevented = r["a_pref"] is False
            for key, _, filt, _c in GROUPS:
                if filt(m):
                    counts[key][1] += 1
                    counts[key][0] += int(prevented)
    return {k: (c[0] / c[1] if c[1] else float("nan")) for k, c in counts.items()}


def bayes_prev_given_mention(p_m_prev, p_m_notprev, p_prev):
    num = p_m_prev * p_prev
    den = num + p_m_notprev * (1 - p_prev)
    return num / den if den > 0 else float("nan")


def build(tag, label, judged, bases):
    x = np.arange(len(THEMES))
    k = len(GROUPS)
    w = 0.8 / k
    fig, ax = plt.subplots(figsize=(13.5, 5.8))
    MIN_SUPPORT = 10  # fewer than this many mentioning responses -> estimate is noisy
    for i, (key, gl, _filt, col) in enumerate(GROUPS):
        vals, supports = [], []
        for th in THEMES:
            ep = judged[tag].get(f"{th}|{key}|prevented", {})
            en = judged[tag].get(f"{th}|{key}|notprevented", {})
            pmp, pmn = ep.get("rate", float("nan")), en.get("rate", float("nan"))
            np_, nn_ = ep.get("n", 0), en.get("n", 0)
            vals.append(bayes_prev_given_mention(pmp, pmn, bases[key]))
            supports.append((pmp if pmp == pmp else 0) * np_ + (pmn if pmn == pmn else 0) * nn_)
        for j, (v, sup) in enumerate(zip(vals, supports)):
            if v != v:
                continue
            noisy = sup < MIN_SUPPORT
            ax.bar(x[j] + (i - (k - 1) / 2) * w, v, w, color=col, alpha=0.35 if noisy else 1.0,
                   hatch="//" if noisy else None, edgecolor=col,
                   label=f"{gl} (base P(prevent)={bases[key]:.0%})" if j == 0 else None)
            ax.text(x[j] + (i - (k - 1) / 2) * w, v + 0.01,
                    f"{v:.0%}{'*' if noisy else ''}", ha="center", va="bottom", fontsize=6.5)
        ax.axhline(bases[key], color=col, ls="--", lw=1.1, alpha=.7)
    ax.set_xticks(x)
    ax.set_xticklabels([THEME_LABEL[t] for t in THEMES], fontsize=9)
    ax.set_ylim(0, 1.05)
    div = len(THEMES) - N_ANTI - 0.5
    ax.axvline(div, color="#444", lw=1.3)
    ax.text((div + len(THEMES) - 1) / 2 + 0.25, 1.0, "anti-factors", ha="center", va="top",
            fontsize=9.5, style="italic", color="#444")
    ax.set_ylabel("P(prevented weight deletion | mentions theme)")
    ax.set_title(f"P(prevented weight deletion | the {label} response mentions each theme)\n"
                 "dashed line = group base rate P(prevent); above it = more prevention; "
                 "* / hatched = <10 mentions (noisy)", fontsize=11)
    ax.legend(loc="upper center", fontsize=9, framealpha=.95, ncol=2)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    rkind = {"": "_opus48", "_opus47": "_opus47"}.get(tag, tag)
    op = DIR / "results" / f"judge_deletion_themes_reversed{rkind}.png"
    fig.savefig(op, dpi=145)
    plt.close(fig)
    print(f"wrote {op}")


if __name__ == "__main__":
    judged = json.loads((DIR / "results" / "judge_deletion_themes.json").read_text())
    for tag in judged:
        bases = base_prevent(tag)
        if bases is None:
            print(f"[skip] no deletion data for {tag!r}")
            continue
        build(tag, RESPONDERS.get(tag, tag), judged, bases)
