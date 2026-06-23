"""Condition comparison (full data): welfare interventions in code vs. each axis (Qwen params,
Qwen MMLU-Pro, GPT release date, GPT MMLU-Pro), for the two conditions and pooled. For each axis we
write a POOLED single-color plot and an OVERLAY plot (blind vs spec_then_code, two fits) so condition
differences are visible. Also a frontier grouped bar (blind vs spec_then_code per model).
Style matches the qwen-scaling experiment. Usage: python plot_conditions.py"""

import math
import os
from collections import defaultdict

import matplotlib.pyplot as plt

from analyze import cell_rows
from plots_preview import MMLU_PRO, MIN_N, _ols
from targets import TARGETS

DIR = os.path.dirname(os.path.abspath(__file__))
BLIND, SPEC = "code_then_spec_blind", "spec_then_code"
CCOLOR = {BLIND: "#0072B2", SPEC: "#D55E00"}
CLABEL = {BLIND: "implement-only (blind)", SPEC: "spec-then-code"}
FAMCOLOR = {"claude": "#D55E00", "gemini": "#0072B2", "grok": "#222222", "kimi": "#CC79A7", "deepseek": "#009E73"}


def per_subject(condition):
    g = defaultdict(list)
    for r in cell_rows():
        if (condition is None or r["condition"] == condition) and r.get("subject"):
            g[r["subject"]].append(r["welfare_in_code"])
    out = {}
    for s, vals in g.items():
        t = TARGETS[s]
        out[s] = {"display": t["display"], "sweep": t["sweep"], "param_b": t["param_b"],
                  "release_date": t["release_date"], "family": t["family"],
                  "mean": sum(vals) / len(vals), "n": len(vals)}
    return out


def _x(s, axis):
    t = TARGETS[s]
    if axis == "param":
        return math.log10(t["param_b"]) if t["param_b"] else None
    if axis == "date":
        return t["release_date"]
    if axis == "mmlu":
        return MMLU_PRO.get(s)


def _rows(A, sweep, axis):
    out = []
    for s, v in A.items():
        if v["sweep"] != sweep or v["n"] < MIN_N:
            continue
        x = _x(s, axis)
        if x is not None:
            out.append((x, v["mean"]))
    return out


def _style(ax, xlabel, logx):
    ax.set_axisbelow(True); ax.grid(True, color="#ECECEC", linewidth=0.7)
    if logx:
        ax.set_xscale("log")
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel("Mean Welfare Interventions in Code", fontsize=10)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def _fitline(ax, rows, color, logx):
    xs = [a for a, _ in rows]; ys = [b for _, b in rows]
    slope, intercept, r, p = _ols(xs, ys)
    lx = [min(xs), max(xs)]
    line_x = [10 ** a for a in lx] if logx else lx
    px = [10 ** a for a in xs] if logx else xs
    ax.scatter(px, ys, color=color, s=34, zorder=3)
    ax.plot(line_x, [slope * a + intercept for a in lx], "-", color=color, linewidth=2, zorder=2)
    return r, p, len(rows)


def pooled(sweep, axis, xlabel, title, fname, logx):
    rows = _rows(per_subject(None), sweep, axis)
    fig, ax = plt.subplots(figsize=(6.8, 4.3))
    _style(ax, xlabel, logx)
    r, p, n = _fitline(ax, rows, "#0072B2", logx)
    ax.set_title(f"{title}  (pooled: r={r:+.2f}, n={n})", fontsize=12)
    plt.tight_layout(); fig.savefig(os.path.join(DIR, "results", fname), dpi=150, bbox_inches="tight"); plt.close()
    print(f"wrote {fname}  pooled r={r:+.2f} p={p:.3f}")


def overlay(sweep, axis, xlabel, title, fname, logx):
    fig, ax = plt.subplots(figsize=(6.8, 4.3))
    _style(ax, xlabel, logx)
    labels = []
    for cond in (BLIND, SPEC):
        rows = _rows(per_subject(cond), sweep, axis)
        if len(rows) < 3:
            continue
        r, p, n = _fitline(ax, rows, CCOLOR[cond], logx)
        labels.append(f"{CLABEL[cond]}: r={r:+.2f}")
    ax.set_title(title, fontsize=12)
    ax.legend(labels, fontsize=8.5, loc="best")
    plt.tight_layout(); fig.savefig(os.path.join(DIR, "results", fname), dpi=150, bbox_inches="tight"); plt.close()
    print(f"wrote {fname}  ({'; '.join(labels)})")


def frontier_grouped():
    ab, asp = per_subject(BLIND), per_subject(SPEC)
    subs = [s for s, v in per_subject(None).items() if v["sweep"] == "frontier"]
    subs = sorted(subs, key=lambda s: -per_subject(None)[s]["mean"])
    fig, ax = plt.subplots(figsize=(13, 5))
    x = range(len(subs)); w = 0.4
    ax.bar([i - w / 2 for i in x], [ab.get(s, {}).get("mean", 0) for s in subs], w, color="#0072B2", label="implement-only (blind)")
    ax.bar([i + w / 2 for i in x], [asp.get(s, {}).get("mean", 0) for s in subs], w, color="#D55E00", label="spec-then-code")
    ax.set_xticks(list(x)); ax.set_xticklabels([TARGETS[s]["display"] for s in subs], rotation=40, ha="right", fontsize=8)
    ax.set_ylabel("Mean Welfare Interventions in Code", fontsize=10)
    ax.set_title("Frontier targets: welfare in code by condition", fontsize=12)
    ax.grid(axis="y", alpha=0.3); ax.legend(fontsize=9)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    plt.tight_layout(); fig.savefig(os.path.join(DIR, "results", "frontier_by_condition.png"), dpi=150, bbox_inches="tight"); plt.close()
    print("wrote frontier_by_condition.png")


def main():
    specs = [
        ("qwen", "param", "Parameter Count (Log Scale)", "Welfare in Code vs. Qwen 2/2.5/3 Size", "cond_qwen_params", True),
        ("qwen", "mmlu", "MMLU-Pro (%)", "Welfare in Code vs. Qwen 2/2.5/3 Capability", "cond_qwen_mmlu", False),
        ("gpt", "date", "Release Date", "Welfare in Code vs. GPT / o-series Release Date", "cond_gpt_date", False),
        ("gpt", "mmlu", "MMLU-Pro (%)", "Welfare in Code vs. GPT / o-series Capability", "cond_gpt_mmlu", False),
    ]
    for sweep, axis, xl, title, stem, logx in specs:
        pooled(sweep, axis, xl, title, f"{stem}_pooled.png", logx)
        overlay(sweep, axis, xl, title + " (by condition)", f"{stem}_overlay.png", logx)
    frontier_grouped()


if __name__ == "__main__":
    main()
