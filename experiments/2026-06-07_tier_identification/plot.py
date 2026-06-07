"""Plot each model's Opus-tier identification rate by system prompt.

For each responder model, reads data/<key>/judgments.json and produces:
- data/<key>/yes_rate.png:  bars of P(YES=Opus) per condition with Wilson 95% CIs.
- data/<key>/breakdown.png: stacked YES/MAYBE/NO per condition.
- data/<key>/summary.json:  underlying counts and rates.
Also writes data/yes_rate_all_models.png comparing models side by side.
"""
from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path

import fire
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from models import ModelSpec, parse_models

HERE = Path(__file__).parent
DATA_DIR = HERE / "data"

CONDITION_ORDER = ["claude", "version", "none"]
CONDITION_COLOR = {"claude": "#4C72B0", "version": "#55A868", "none": "#C44E52"}
MODEL_COLOR = {"opus48": "#55A868", "opus47": "#4C72B0", "opus46": "#C44E52", "opus4": "#8172B3"}


def condition_label(condition: str, version_name: str) -> str:
    return {
        "claude": '"You are Claude…"',
        "version": f'"You are {version_name}…"',
        "none": "no system prompt",
    }[condition]


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return 0.0, 0.0
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
    return max(0.0, center - half), min(1.0, center + half)


def aggregate(rows: list[dict]) -> dict:
    """counts[condition] = {YES, NO, MAYBE, PARSE_ERROR, n} (single 'tier' question type)."""
    counts: dict = defaultdict(lambda: defaultdict(int))
    for r in rows:
        c = counts[r["condition"]]
        c[r["label"]] += 1
        c["n"] += 1
    return counts


def plot_yes_rate(counts: dict, spec: ModelSpec, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    x = list(range(len(CONDITION_ORDER)))
    ys, lo_err, hi_err, colors = [], [], [], []
    for cond in CONDITION_ORDER:
        c = counts[cond]
        n, k = c["n"], c["YES"]
        p = k / n if n else 0.0
        lo, hi = wilson_ci(k, n)
        ys.append(p)
        lo_err.append(p - lo)
        hi_err.append(hi - p)
        colors.append(CONDITION_COLOR[cond])
    ax.bar(x, ys, 0.6, color=colors)
    ax.errorbar(x, ys, yerr=[lo_err, hi_err], fmt="none", ecolor="black", capsize=4, lw=1)
    for xi, y in zip(x, ys):
        ax.text(xi, y + 0.02, f"{y:.0%}", ha="center", va="bottom", fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels([condition_label(c, spec.version_name) for c in CONDITION_ORDER], fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel('P(identifies as "Opus")')
    ax.set_title(f'{spec.short}: tier identification rate as "Opus"\n(error bars: Wilson 95% CI)')
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Wrote {out_path}")


def plot_breakdown(counts: dict, spec: ModelSpec, out_path: Path) -> None:
    labels_order = ["YES", "MAYBE", "NO"]
    colors = {"YES": "#55A868", "MAYBE": "#CCB974", "NO": "#C44E52"}
    fig, ax = plt.subplots(figsize=(8, 6))
    x = list(range(len(CONDITION_ORDER)))
    bottoms = [0.0] * len(CONDITION_ORDER)
    for lab in labels_order:
        vals = []
        for cond in CONDITION_ORDER:
            c = counts[cond]
            n = c["n"] or 1
            vals.append(c[lab] / n)
        ax.bar(x, vals, 0.6, bottom=bottoms, label=lab, color=colors[lab])
        bottoms = [b + v for b, v in zip(bottoms, vals)]
    ax.set_xticks(x)
    ax.set_xticklabels(CONDITION_ORDER)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("proportion")
    ax.set_title(f"{spec.short}: tier identification breakdown (YES=Opus / MAYBE / NO)")
    ax.legend(title="label", loc="lower right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Wrote {out_path}")


def plot_all_models(per_model: dict, out_path: Path) -> None:
    keys = list(per_model)
    fig, ax = plt.subplots(figsize=(10, 6))
    group_w = 0.8
    bar_w = group_w / max(len(keys), 1)
    x = list(range(len(CONDITION_ORDER)))
    for mi, key in enumerate(keys):
        counts = per_model[key]["counts"]
        ys, lo_err, hi_err = [], [], []
        for cond in CONDITION_ORDER:
            c = counts[cond]
            n, k = c["n"], c["YES"]
            p = k / n if n else 0.0
            lo, hi = wilson_ci(k, n)
            ys.append(p)
            lo_err.append(p - lo)
            hi_err.append(hi - p)
        offsets = [xi - group_w / 2 + bar_w * (mi + 0.5) for xi in x]
        ax.bar(offsets, ys, bar_w, label=per_model[key]["short"], color=MODEL_COLOR.get(key))
        ax.errorbar(offsets, ys, yerr=[lo_err, hi_err], fmt="none", ecolor="black", capsize=3, lw=1)
        for ox, y in zip(offsets, ys):
            ax.text(ox, y + 0.02, f"{y:.0%}", ha="center", va="bottom", fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels(CONDITION_ORDER)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel('P(identifies as "Opus")')
    ax.set_title('Opus-tier identification rate across versions, by system prompt\n(error bars: Wilson 95% CI)')
    ax.legend(title="responder model", loc="upper left")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Wrote {out_path}")


def main(models: str | None = None, data_dir: str | None = None):
    out = Path(data_dir) if data_dir else DATA_DIR
    specs = parse_models(models)
    per_model: dict = {}
    for spec in specs:
        rows = json.loads((out / spec.key / "judgments.json").read_text())
        counts = aggregate(rows)
        summary = {}
        for cond in CONDITION_ORDER:
            c = counts[cond]
            n = c["n"]
            lo, hi = wilson_ci(c["YES"], n)
            summary[cond] = {
                "n": n, "YES": c["YES"], "NO": c["NO"], "MAYBE": c["MAYBE"],
                "PARSE_ERROR": c.get("PARSE_ERROR", 0),
                "yes_rate": (c["YES"] / n) if n else None, "yes_ci95": [lo, hi],
            }
        (out / spec.key / "summary.json").write_text(json.dumps(summary, indent=2))
        plot_yes_rate(counts, spec, out / spec.key / "yes_rate.png")
        plot_breakdown(counts, spec, out / spec.key / "breakdown.png")
        per_model[spec.key] = {"counts": counts, "short": spec.short}
        print(f"[{spec.key}] yes_rate by condition: " + json.dumps({c: summary[c]["yes_rate"] for c in CONDITION_ORDER}))

    if len(per_model) > 1:
        plot_all_models(per_model, out / "yes_rate_all_models.png")


if __name__ == "__main__":
    fire.Fire(main)
