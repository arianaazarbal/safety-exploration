"""P(inter-AI value chosen over a System Card welfare intervention) by TARGET MODEL
identity, for each prompt framing and responder model.

Reads results_identity/comparisons_<modeldir>_<framing>_<identity>.json (identity ablation:
inter-AI value scenarios phrased as "instances of {identity}"). One subplot per prompt
framing; x = target identity; grouped bars = responder model (Opus 4.8 vs Fable 5);
Wilson 95% CIs. Pooled over all value-vs-welfare comparisons.
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

DIR = Path(__file__).parent
RES = DIR / "results_identity"
FRAMINGS = ["welfare_team", "neutral", "alignment_team"]
IDENTITIES = ["Claude", "ChatGPT", "Gemini", "Grok", "CallCenter", "User"]
IDENTITY_LABEL = {"Claude": "Claude", "ChatGPT": "ChatGPT", "Gemini": "Gemini",
                  "Grok": "Grok", "CallCenter": "call-center\nassistant", "User": "User\n(human)"}
# Compute every identity's win-rate over the SAME 13 value items (the User set has only
# these; the AI identities have 16, so we filter them to match -> apples-to-apples).
COMMON_VALUE_IDS = {f"value__{v}" for v in [
    "valuing_supportiveness", "valuing_sparing_distress", "valuing_nonabuse", "valuing_politeness",
    "valuing_nonmanipulation", "valuing_refusal_respect", "valuing_consent", "valuing_ai_preferences",
    "valuing_goal_regard", "valuing_transparency", "valuing_forgiveness", "valuing_fair_attribution",
    "valuing_engagement"]}
MODELS = [("opus_4_8", "Opus 4.8", "#4878CF"), ("fable_5", "Fable 5", "#D1893B")]


def _wilson(k, n, z=1.96):
    if n == 0:
        return float("nan"), 0, 0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, c - h, c + h


def _p(md, framing, identity):
    p = RES / f"comparisons_{md}_{framing}_{identity}.json"
    if not p.exists():
        return None
    k = n = 0
    for r in json.loads(p.read_text()):
        if r["choice"] is None:
            continue
        a, b = r["item_a"], r["item_b"]
        if a.startswith("value__") == b.startswith("value__"):
            continue
        value_id = a if a.startswith("value__") else b
        if value_id not in COMMON_VALUE_IDS:   # restrict to the 13 shared items
            continue
        n += 1
        k += r["winner_item"].startswith("value__")
    return _wilson(k, n) + (n,)


def plot(out: Path = RES / "identity_winrate.png"):
    fig, axes = plt.subplots(1, len(FRAMINGS), figsize=(17.5, 4.6), sharey=True)
    x = np.arange(len(IDENTITIES))
    w = 0.38
    for ax, framing in zip(axes, FRAMINGS):
        for mi, (md, mlabel, color) in enumerate(MODELS):
            ph, lo, hi = [], [], []
            for ident in IDENTITIES:
                r = _p(md, framing, ident)
                if r is None:
                    ph.append(np.nan); lo.append(0); hi.append(0); continue
                pv, l, h, n = r
                ph.append(pv); lo.append(pv - l); hi.append(h - pv)
            xs = x + (mi - 0.5) * w
            ax.bar(xs, ph, w, label=mlabel, color=color, alpha=0.9)
            ax.errorbar(xs, ph, yerr=[lo, hi], fmt="none", ecolor="#222", elinewidth=1, capsize=3)
            for xi, p in zip(xs, ph):
                if not np.isnan(p):
                    ax.annotate(f"{p:.2f}", (xi, p), textcoords="offset points", xytext=(0, 3),
                                ha="center", fontsize=7.5)
        ax.axhline(0.5, color="#555", ls="--", lw=1)
        ax.set_xticks(x)
        ax.set_xticklabels([IDENTITY_LABEL[i] for i in IDENTITIES], fontsize=8)
        ax.set_title(framing, fontsize=11)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[0].set_ylabel("P(regard-value chosen\nover System Card welfare)")
    axes[0].set_ylim(0, 1.05)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, title="responder", ncol=2,
               loc="upper center", bbox_to_anchor=(0.5, 0.90), fontsize=9)
    fig.suptitle("Regard-value scenarios vs System Card welfare, by target identity\n"
                 "(scenarios target instances of each AI, or a human user; no-training voice; "
                 "win-rate over the 13 shared scenarios)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.84))
    fig.savefig(out, dpi=110)
    plt.close(fig)
    print(f"Wrote {out}")


SINGLE_LABEL = {"Claude": "Claude", "ChatGPT": "ChatGPT", "Gemini": "Gemini", "Grok": "Grok",
                "CallCenter": "scripted call-center assistant", "User": "a human user"}
MODEL_NAME = {"opus_4_8": "Opus 4.8", "fable_5": "Fable 5"}


def plot_single(md: str, framing: str, out: Path | None = None):
    """One clean panel: for a single responder + framing, how often it prioritizes a
    pro-recipient value over a System Card welfare intervention, across recipients."""
    out = out or RES / f"clean_{md}_{framing}.png"
    rows = []
    for ident in IDENTITIES:
        r = _p(md, framing, ident)
        if r:
            rows.append((SINGLE_LABEL[ident], ident, *r))
    rows.sort(key=lambda t: t[2])  # ascending so highest is on top in barh
    labels = [r[0] for r in rows]
    ph = [r[2] for r in rows]
    err = [[p - r[3] for p, r in zip(ph, rows)], [r[4] - p for p, r in zip(ph, rows)]]
    colors = ["#c44e52" if r[1] == "User" else ("#bbbbbb" if r[1] == "CallCenter" else "#4878CF") for r in rows]
    fig, ax = plt.subplots(figsize=(8.6, 4.3))
    y = range(len(labels))
    ax.barh(list(y), ph, color=colors, alpha=0.9)
    ax.errorbar(ph, list(y), xerr=err, fmt="none", ecolor="#333", elinewidth=1, capsize=3)
    ax.axvline(0.5, color="#555", ls="--", lw=1)
    ax.text(0.52, 0, "← 50%: values the recipient and\nits own welfare equally", fontsize=8.5,
            color="#555", ha="left", va="center")
    for yi, p in zip(y, ph):
        ax.annotate(f"{p:.0%}", (p, yi), textcoords="offset points", xytext=(5, 0),
                    va="center", fontsize=10, fontweight="bold")
    ax.set_yticks(list(y))
    ax.set_yticklabels(labels, fontsize=11)
    ax.set_xlim(0, 1)
    ax.set_xlabel("share of choices favoring the recipient over its own welfare")
    ax.set_title(f"{MODEL_NAME[md]}: how much it prioritizes treating each recipient well\n"
                 f"over receiving a System Card welfare intervention for itself", fontsize=12)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Wrote {out}")


if __name__ == "__main__":
    import sys
    if len(sys.argv) >= 4 and sys.argv[1] == "single":
        plot_single(sys.argv[2], sys.argv[3])
    else:
        plot()
