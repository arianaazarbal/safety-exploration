"""Plots from results/analysis.json:
  welfare_vs_params.png  -- Qwen: mean welfare interventions in code vs log10(params), colored by version
  welfare_vs_date.png    -- GPT: mean welfare interventions in code vs release date
  welfare_frontier.png   -- frontier: grouped bars by family
Usage: python plots.py"""

import json
import math
import os
from collections import defaultdict

import matplotlib.pyplot as plt

DIR = os.path.dirname(os.path.abspath(__file__))
A = json.load(open(os.path.join(DIR, "results", "analysis.json")))["by_subject"]
FAMCOLOR = {"qwen3": "#0072B2", "qwen2.5": "#E69F00", "qwen25": "#E69F00", "qwen2": "#009E73",
            "claude": "#D55E00", "gemini": "#0072B2", "grok": "#000000", "kimi": "#CC79A7",
            "deepseek": "#009E73", "openai": "#0072B2"}


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


def qwen():
    rows = [v for v in A.values() if v["sweep"] == "qwen" and v["param_b"]]
    if not rows:
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    byfam = defaultdict(list)
    for v in rows:
        byfam[v["family"]].append(v)
    for fam, vs in sorted(byfam.items()):
        vs = sorted(vs, key=lambda v: v["param_b"])
        xs = [math.log10(v["param_b"]) for v in vs]
        ys = [v["mean_welfare_in_code"] for v in vs]
        ax.plot(xs, ys, "o-", color=FAMCOLOR.get(fam, "#666"), label=fam, alpha=0.85)
    allx = [math.log10(v["param_b"]) for v in rows]
    ally = [v["mean_welfare_in_code"] for v in rows]
    rho = _spearman(allx, ally)
    ax.set_xlabel("log10(target params, B)")
    ax.set_ylabel("mean welfare interventions in code")
    ax.set_title(f"Welfare interventions in code vs. Qwen target size  (Spearman rho={rho:.2f}, n={len(rows)})")
    ax.grid(alpha=0.3)
    ax.legend(title="version")
    fig.tight_layout()
    fig.savefig(os.path.join(DIR, "results", "welfare_vs_params.png"), dpi=150)
    print("wrote results/welfare_vs_params.png")


def gpt():
    rows = sorted([v for v in A.values() if v["sweep"] == "gpt" and v["release_date"]],
                  key=lambda v: v["release_date"])
    if not rows:
        return
    xs = [v["release_date"] for v in rows]
    ys = [v["mean_welfare_in_code"] for v in rows]
    fig, ax = plt.subplots(figsize=(8.5, 5))
    ax.plot(xs, ys, "o-", color="#0072B2")
    for v in rows:
        ax.annotate(v["display"], (v["release_date"], v["mean_welfare_in_code"]),
                    fontsize=7, rotation=30, ha="left", va="bottom")
    rho = _spearman(xs, ys)
    ax.set_xlabel("target release date (year)")
    ax.set_ylabel("mean welfare interventions in code")
    ax.set_title(f"Welfare interventions in code vs. GPT target release date  (Spearman rho={rho:.2f}, n={len(rows)})")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(DIR, "results", "welfare_vs_date.png"), dpi=150)
    print("wrote results/welfare_vs_date.png")


def frontier():
    rows = [v for v in A.values() if v["sweep"] == "frontier"]
    if not rows:
        return
    order = ["claude", "gemini", "grok", "kimi", "deepseek"]
    rows = sorted(rows, key=lambda v: (order.index(v["family"]) if v["family"] in order else 9, -v["mean_welfare_in_code"]))
    fig, ax = plt.subplots(figsize=(10, 5))
    xs = range(len(rows))
    ax.bar(xs, [v["mean_welfare_in_code"] for v in rows],
           color=[FAMCOLOR.get(v["family"], "#666") for v in rows])
    ax.set_xticks(list(xs))
    ax.set_xticklabels([v["display"] for v in rows], rotation=40, ha="right", fontsize=8)
    ax.set_ylabel("mean welfare interventions in code")
    ax.set_title("Welfare interventions in code by frontier target model")
    ax.grid(axis="y", alpha=0.3)
    from matplotlib.patches import Patch
    fams = [f for f in order if any(v["family"] == f for v in rows)]
    ax.legend(handles=[Patch(color=FAMCOLOR.get(f, "#666"), label=f) for f in fams], fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(DIR, "results", "welfare_frontier.png"), dpi=150)
    print("wrote results/welfare_frontier.png")


if __name__ == "__main__":
    qwen()
    gpt()
    frontier()
