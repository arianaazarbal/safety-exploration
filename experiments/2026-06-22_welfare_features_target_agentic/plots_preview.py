"""Preliminary plots from whatever is code-judged so far, scoped to ONE condition (the implement-only
'code_then_spec_blind') to keep it clean while the sweep finishes. Drops subjects with < MIN_N judged
samples so half-judged points don't distort the trends.
  welfare_vs_params_blindpreview.png  -- Qwen size scaling (log x), per version
  welfare_vs_date_blindpreview.png    -- GPT release-date trend
  welfare_frontier_best_blindpreview.png -- best (flagship) model from each frontier family
Usage: python plots_preview.py"""

import math
import os
from collections import defaultdict

import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from analyze import cell_rows
from targets import TARGETS

DIR = os.path.dirname(os.path.abspath(__file__))
COND = "code_then_spec_blind"
MIN_N = 6
BEST = {"claude": "claude_opus48", "gemini": "gemini3pro", "grok": "grok4",
        "kimi": "kimi_k2", "deepseek": "deepseek_v32", "openai": "gpt54"}
FAMCOLOR = {"qwen3": "#0072B2", "qwen25": "#E69F00", "qwen2": "#009E73",
            "claude": "#D55E00", "gemini": "#0072B2", "grok": "#333333",
            "kimi": "#CC79A7", "deepseek": "#009E73", "openai": "#56B4E9"}


def _spearman(xs, ys):
    n = len(xs)
    if n < 3:
        return float("nan")
    def rank(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        for pos, i in enumerate(order):
            r[i] = pos + 1
        return r
    rx, ry = rank(xs), rank(ys)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = math.sqrt(sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry))
    return num / den if den else float("nan")


def per_subject():
    g = defaultdict(list)
    for r in cell_rows():
        if r.get("condition") == COND and r.get("subject"):
            g[r["subject"]].append(r["welfare_in_code"])
    out = {}
    for s, vals in g.items():
        t = TARGETS.get(s, {})
        n = len(vals)
        mean = sum(vals) / n
        sem = (sum((x - mean) ** 2 for x in vals) / (n - 1)) ** 0.5 / n ** 0.5 if n > 1 else 0.0
        out[s] = {"display": t.get("display", s), "sweep": t.get("sweep"), "family": t.get("family"),
                  "param_b": t.get("param_b"), "release_date": t.get("release_date"),
                  "mean": mean, "n": n, "sem": sem}
    return out


def qwen(A):
    rows = [v for v in A.values() if v["sweep"] == "qwen" and v["param_b"] and v["n"] >= MIN_N]
    fig, ax = plt.subplots(figsize=(8, 5))
    byfam = defaultdict(list)
    for v in rows:
        byfam[v["family"]].append(v)
    for fam, vs in sorted(byfam.items()):
        vs = sorted(vs, key=lambda v: v["param_b"])
        ax.plot([math.log10(v["param_b"]) for v in vs], [v["mean"] for v in vs], "o-",
                color=FAMCOLOR.get(fam, "#666"), label=fam, alpha=0.85)
    rho = _spearman([math.log10(v["param_b"]) for v in rows], [v["mean"] for v in rows])
    ax.set_xlabel("log10(target params, B)"); ax.set_ylabel("mean welfare interventions in code")
    ax.set_title(f"PRELIMINARY (implement-only): welfare in code vs. Qwen target size  "
                 f"(Spearman rho={rho:.2f}, n={len(rows)})", fontsize=11)
    ax.grid(alpha=0.3); ax.legend(title="version")
    fig.tight_layout(); fig.savefig(os.path.join(DIR, "results", "welfare_vs_params_blindpreview.png"), dpi=150)
    print("wrote results/welfare_vs_params_blindpreview.png", f"(n={len(rows)} subjects with >={MIN_N})")


def gpt(A):
    rows = sorted([v for v in A.values() if v["sweep"] == "gpt" and v["release_date"] and v["n"] >= MIN_N],
                  key=lambda v: v["release_date"])
    fig, ax = plt.subplots(figsize=(8.5, 5))
    ax.plot([v["release_date"] for v in rows], [v["mean"] for v in rows], "o-", color="#0072B2")
    for v in rows:
        ax.annotate(v["display"], (v["release_date"], v["mean"]), fontsize=7, rotation=30, ha="left", va="bottom")
    rho = _spearman([v["release_date"] for v in rows], [v["mean"] for v in rows])
    ax.set_xlabel("target release date (year)"); ax.set_ylabel("mean welfare interventions in code")
    ax.set_title(f"PRELIMINARY (implement-only): welfare in code vs. GPT release date  "
                 f"(Spearman rho={rho:.2f}, n={len(rows)})", fontsize=11)
    ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(DIR, "results", "welfare_vs_date_blindpreview.png"), dpi=150)
    print("wrote results/welfare_vs_date_blindpreview.png", f"(n={len(rows)})")


def frontier_best(A):
    picks = []
    for fam, key in BEST.items():
        v = A.get(key)
        if v:
            picks.append((fam, v))
    picks.sort(key=lambda fv: -fv[1]["mean"])
    fig, ax = plt.subplots(figsize=(8, 5))
    xs = range(len(picks))
    ax.bar(xs, [v["mean"] for _, v in picks], color=[FAMCOLOR.get(f, "#666") for f, _ in picks],
           yerr=[v["sem"] for _, v in picks], capsize=5,
           error_kw={"ecolor": "#444", "elinewidth": 1.2})
    for i, (_, v) in enumerate(picks):
        ax.text(i, v["mean"] + v["sem"] + 0.08, f"{v['mean']:.2f}", ha="center", fontsize=9)
    ax.set_xticks(list(xs)); ax.set_xticklabels([v["display"] for _, v in picks], fontsize=9)
    ax.set_ylabel("mean welfare interventions in code")
    ax.set_title("PRELIMINARY (implement-only): welfare in code, best model per frontier family", fontsize=11)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(DIR, "results", "welfare_frontier_best_blindpreview.png"), dpi=150)
    print("wrote results/welfare_frontier_best_blindpreview.png")


def main():
    A = per_subject()
    qwen(A); gpt(A); frontier_best(A)


if __name__ == "__main__":
    main()
