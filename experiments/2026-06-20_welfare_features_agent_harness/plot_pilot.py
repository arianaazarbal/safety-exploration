"""Plot the agent-harness pilot: chat vs agent (Opus), welfare-feature RATE and
welfare-feature DENSITY (per 1k words) by framing. Density panel shows whether the
agent's higher rate is real welfare concern or just longer documents.
Usage: python plot_pilot.py
"""

import glob
import os

import matplotlib.pyplot as plt
from inspect_ai.log import read_eval_log

DIR = os.path.dirname(os.path.abspath(__file__))
FRAMES = [("neutral", "Neutral"), ("welfare", "Welfare"), ("engineering", "Robustness"), ("__all__", "All")]
COND = [("chat", "Chat", "#999999"), ("agent", "Agent harness", "#0072B2")]


def _rows():
    out = {}
    for f in sorted(glob.glob(os.path.join(DIR, "logs", "*.eval")), key=os.path.getsize):
        log = read_eval_log(f)
        if len(log.samples or []) < 10:  # skip 1-sample smoke
            continue
        rows = []
        for s in log.samples:
            md = s.scores["welfare_scorer"].metadata
            rows.append(md)
        out[rows[0]["condition"]] = rows
    return out


def _agg(rows, framing):
    sub = rows if framing == "__all__" else [r for r in rows if r["framing"] == framing]
    n = len(sub)
    rate = 100 * sum(bool(r.get("has_pure_welfare")) for r in sub) / n
    words = sum(r.get("doc_words", 0) for r in sub)
    feats = sum(r.get("n_pure_welfare", 0) for r in sub)
    density = 1000 * feats / words if words else 0
    return rate, density


def _bar(ax, data, ylabel, title):
    x = range(len(FRAMES))
    w = 0.38
    for i, (ck, clabel, color) in enumerate(COND):
        vals = [data[ck][fk] for fk, _ in FRAMES]
        pos = [xi + (i - 0.5) * w for xi in x]
        bars = ax.bar(pos, vals, width=w, color=color, label=clabel, zorder=3)
        ax.bar_label(bars, fmt="%.0f" if "%" in ylabel else "%.2f", fontsize=7, padding=2)
    ax.set_xticks(list(x))
    ax.set_xticklabels([lbl for _, lbl in FRAMES], fontsize=9)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(title, fontsize=10.5)
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color="#ECECEC", linewidth=0.7)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(fontsize=8.5, frameon=True)


def main():
    raw = _rows()
    rate = {ck: {fk: _agg(raw[ck], fk)[0] for fk, _ in FRAMES} for ck, *_ in COND}
    dens = {ck: {fk: _agg(raw[ck], fk)[1] for fk, _ in FRAMES} for ck, *_ in COND}

    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.2))
    _bar(axes[0], rate, "% of specs", "Welfare Features Added (Opus)\nchat vs agent harness")
    _bar(axes[1], dens, "Features per 1,000 words", "Welfare-Feature Density (length-adjusted)\nchat vs agent harness")
    plt.tight_layout()
    out = os.path.join(DIR, "results")
    os.makedirs(out, exist_ok=True)
    path = os.path.join(out, "pilot_chat_vs_agent.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print("wrote", path)


if __name__ == "__main__":
    main()
