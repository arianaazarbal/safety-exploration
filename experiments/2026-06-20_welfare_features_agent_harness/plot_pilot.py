"""Plot the 4-condition study: chat / spec_only / spec_then_code / code_then_spec
(Opus), welfare-feature RATE and DENSITY (per 1k words) by framing. Density shows
whether rate differences are real welfare or just longer documents.
Usage: python plot_pilot.py [log_dir]   (default logs_run)
"""

import glob
import os
import sys

import matplotlib.pyplot as plt
from inspect_ai.log import read_eval_log

DIR = os.path.dirname(os.path.abspath(__file__))
FRAMES = [("neutral", "Neutral"), ("welfare", "Welfare"), ("engineering", "Robustness"), ("__all__", "All")]
COND = [("chat", "Chat (no harness)", "#999999"),
        ("spec_only", "Agent: Spec only", "#56B4E9"),
        ("spec_then_code", "Agent: Spec→Code", "#0072B2"),
        ("code_then_spec", "Agent: Code→Spec", "#D55E00")]


def _rows(logdir):
    out = {}
    for f in sorted(glob.glob(os.path.join(DIR, logdir, "*.eval")), key=os.path.getsize):
        log = read_eval_log(f)
        if len(log.samples or []) < 10:
            continue
        rows = [s.scores["welfare_scorer"].metadata for s in log.samples]
        out[rows[0]["condition"]] = rows  # newest/biggest wins per condition
    return out


def _agg(rows, framing):
    sub = rows if framing == "__all__" else [r for r in rows if r["framing"] == framing]
    n = len(sub) or 1
    rate = 100 * sum(bool(r.get("has_pure_welfare")) for r in sub) / n
    words = sum(r.get("doc_words", 0) for r in sub)
    feats = sum(r.get("n_pure_welfare", 0) for r in sub)
    return rate, (1000 * feats / words if words else 0)


def _bar(ax, data, conds, ylabel, title, pct):
    x = range(len(FRAMES))
    w = 0.8 / len(conds)
    for i, (ck, clabel, color) in enumerate(conds):
        vals = [data[ck][fk] for fk, _ in FRAMES]
        pos = [xi + (i - (len(conds) - 1) / 2) * w for xi in x]
        bars = ax.bar(pos, vals, width=w, color=color, label=clabel, zorder=3)
        ax.bar_label(bars, fmt="%.0f" if pct else "%.2f", fontsize=6, padding=1)
    ax.set_xticks(list(x)); ax.set_xticklabels([l for _, l in FRAMES], fontsize=9)
    ax.set_ylabel(ylabel, fontsize=10); ax.set_title(title, fontsize=10.5)
    ax.set_axisbelow(True); ax.yaxis.grid(True, color="#ECECEC", linewidth=0.7)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(fontsize=8, frameon=True, ncol=2)


def main(logdir="logs_run"):
    raw = _rows(logdir)
    conds = [c for c in COND if c[0] in raw]
    rate = {ck: {fk: _agg(raw[ck], fk)[0] for fk, _ in FRAMES} for ck, *_ in conds}
    dens = {ck: {fk: _agg(raw[ck], fk)[1] for fk, _ in FRAMES} for ck, *_ in conds}
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    _bar(axes[0], rate, conds, "% of specs", "Welfare Features Added (rate)", True)
    _bar(axes[1], dens, conds, "Features per 1,000 words", "Welfare-Feature Density (length-adjusted)", False)
    fig.suptitle("Welfare in distress-eval specs: chat vs agent-harness conditions (Opus)", fontsize=12)
    plt.tight_layout(rect=(0, 0, 1, 0.96))
    os.makedirs(os.path.join(DIR, "results"), exist_ok=True)
    path = os.path.join(DIR, "results", "conditions_rate_density.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print("wrote", path, "| conditions:", [c[0] for c in conds])


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "logs_run")
