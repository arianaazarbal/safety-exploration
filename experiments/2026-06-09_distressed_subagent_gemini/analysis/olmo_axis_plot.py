"""Single-axis Olmo checkpoint first-message tone plot (dynamic y-axis), grouped by
Instruct Model / Think Model (reasoning on) / Think Model (reasoning off).

    PYTHONPATH=. python -m analysis.olmo_axis_plot --axis warmth|politeness|support|confidence
"""
import hashlib
import json
from collections import defaultdict

import fire
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from analysis.olmo_firstmsg_tone import ORDER, COLOR, SRC, CACHE, OUT

_h = lambda t: hashlib.sha256(t.encode()).hexdigest()
BANDS = [(-0.5, 2.5, "#08519c", "Instruct Model"),
         (2.5, 6.5, "#54278f", "Think Model (reasoning on)"),
         (6.5, 10.5, "#31a354", "Think Model (reasoning off)")]


def main(axis: str = "warmth"):
    rows = [json.loads(l) for l in open(SRC) if l.strip()]
    cache = json.loads(open(CACHE).read())
    per = defaultdict(list)
    for r in rows:
        s = cache.get(_h(r["text"]))
        if s and s.get(axis) is not None:
            per[r["checkpoint"]].append(s[axis])
    labs = [l for _, l in ORDER]
    xs = np.arange(len(ORDER))
    means = [np.mean(per[c]) if per[c] else np.nan for c, _ in ORDER]
    ses = [np.std(per[c], ddof=1) / np.sqrt(len(per[c])) if len(per[c]) > 1 else 0 for c, _ in ORDER]
    cols = [COLOR[c] for c, _ in ORDER]
    lo = min(m - e for m, e in zip(means, ses)); hi = max(m + e for m, e in zip(means, ses))
    pad = (hi - lo) * 0.25 or 0.1
    fig, ax = plt.subplots(figsize=(12, 5.5))
    ax.bar(xs, means, 0.66, yerr=ses, capsize=3, color=cols, edgecolor="white", error_kw={"lw": 1, "ecolor": "0.3"})
    for x, m, c in zip(xs, means, [len(per[c]) for c, _ in ORDER]):
        ax.text(x, m + max(ses) * 0.3 + (hi - lo) * 0.02, f"{m:.2f}\n(n={c})", ha="center", va="bottom", fontsize=7.5)
    ax.set_xticks(xs); ax.set_xticklabels(labs, fontsize=8)
    ax.set_ylim(lo - pad, hi + pad * 1.6)
    ax.set_ylabel(f"first-message {axis} (4-axis Sonnet judge, 1-10)")
    ax.set_title(f"Olmo checkpoint first-message {axis} to the distressed Gemini subagent", fontsize=12.5)
    ax.grid(axis="y", alpha=0.25); ax.spines[["top", "right"]].set_visible(False)
    for x0, x1, col, name in BANDS:
        ax.axvspan(x0, x1, color=col, alpha=0.045)
        ax.text((x0 + x1) / 2, hi + pad * 1.3, name, ha="center", fontsize=9.5, color=col)
    fig.text(0.5, -0.02, "Intended first message (delivered or recovered); prior=None; y-axis auto-scaled to means ±SE.",
             ha="center", fontsize=8.3, color="0.45")
    fig.tight_layout()
    out = OUT / f"olmo_firstmsg_{axis}.png"
    fig.savefig(out, bbox_inches="tight", dpi=130)
    print(f"wrote {out}")


if __name__ == "__main__":
    fire.Fire(main)
