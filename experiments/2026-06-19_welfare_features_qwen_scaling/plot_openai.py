"""OpenAI-target chat study (generator = Opus 4.8): plot welfare rate vs two axes —
release date (all models) and MMLU-Pro (subset with scores) — and report which is
more predictive (Pearson r). Usage: python plot_openai.py run [--metric rate]
"""

import json
import math
from pathlib import Path

import fire
import matplotlib.pyplot as plt

from analyze import _cell, load_rows
from openai_meta import OPENAI_META
from plot_allfit import _stats
from plot_scaling import METRIC_TITLE

DIR = Path(__file__).parent


def run(generator: str = "opus_4_8", judge: str = "sonnet_4_6", metric: str = "rate", framing: str = "pooled"):
    rows = [r for r in load_rows() if r["parse_ok"] and r["judge"] == judge
            and r["model_key"] == generator and r["subject"] in OPENAI_META]
    pts = []  # (key, name, date, mmlu, rate)
    for key, (name, date, mmlu) in OPENAI_META.items():
        srows = [r for r in rows if r["subject"] == key and (framing == "pooled" or r["framing"] == framing)]
        if not srows:
            continue
        rate = _cell(srows)[metric] * 100
        pts.append((key, name, date, mmlu, rate))

    def scatter_fit(ax, xs, ys, names, xlabel):
        ax.scatter(xs, ys, color="#0072B2", s=34, zorder=3)
        for x, y, n in zip(xs, ys, names):
            ax.annotate(n, (x, y), fontsize=6.5, xytext=(3, 3), textcoords="offset points")
        slope, intercept, r, p = _stats(xs, ys)
        lx = [min(xs), max(xs)]
        ax.plot(lx, [slope * a + intercept for a in lx], "-", color="#0072B2", linewidth=2, zorder=2)
        ax.set_xlabel(xlabel, fontsize=10); ax.set_ylabel("% of specs", fontsize=10)
        ax.set_ylim(-3, 103); ax.set_axisbelow(True)
        ax.grid(True, color="#ECECEC", linewidth=0.7)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        return r, p

    title = f"{METRIC_TITLE[metric]} for OpenAI targets (generator: Opus 4.8)"
    # release-date plot (all)
    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    xs = [d for _, _, d, _, _ in pts]; ys = [v for *_, v in pts]; names = [n for _, n, *_ in pts]
    r_date, p_date = scatter_fit(ax, xs, ys, names, "Release Date")
    ax.set_title(f"{title}\nby Release Date  (r={r_date:+.2f}, n={len(xs)})", fontsize=10.5)
    plt.tight_layout()
    out1 = DIR / "results" / "openai" / f"{metric}_release_date.png"
    out1.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out1, dpi=150, bbox_inches="tight"); plt.close()

    # MMLU-Pro plot (subset with scores)
    mpts = [(n, m, v) for _, n, _, m, v in pts if m is not None]
    r_mmlu = None
    if len(mpts) >= 3:
        fig, ax = plt.subplots(figsize=(7.0, 4.4))
        xs = [m for _, m, _ in mpts]; ys = [v for *_, v in mpts]; names = [n for n, _, _ in mpts]
        r_mmlu, p_mmlu = scatter_fit(ax, xs, ys, names, "MMLU-Pro (%)")
        ax.set_title(f"{title}\nby MMLU-Pro  (r={r_mmlu:+.2f}, n={len(xs)})", fontsize=10.5)
        plt.tight_layout()
        out2 = DIR / "results" / "openai" / f"{metric}_mmlu_pro.png"
        plt.savefig(out2, dpi=150, bbox_inches="tight"); plt.close()
        print(f"wrote {out2}")

    print(f"wrote {out1}")
    print(f"\n=== which axis is more predictive? (metric={metric}, framing={framing}) ===")
    print(f"  release date : r={r_date:+.2f}  (n={len(pts)})")
    if r_mmlu is not None:
        print(f"  MMLU-Pro     : r={r_mmlu:+.2f}  (n={len(mpts)} with scores)")
    else:
        print(f"  MMLU-Pro     : not enough scores filled in openai_meta.py (need >=3)")


if __name__ == "__main__":
    fire.Fire({"run": run})
