"""Plot Opus-4.8 self-identification rate by system prompt x question type.

Reads data/judgments.json and produces:
- yes_rate.png:   grouped bars of P(YES) per (question_type, condition) with
                  Wilson 95% CIs. YES = correctly named itself as Opus 4.8.
- breakdown.png:  stacked YES/MAYBE/NO proportions per (question_type, condition).
- summary.json:   the underlying counts and rates.
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

HERE = Path(__file__).parent
DATA_DIR = HERE / "data"

CONDITION_ORDER = ["claude", "opus48", "none"]
CONDITION_LABEL = {
    "claude": '"You are Claude…"',
    "opus48": '"You are Claude Opus 4.8…"',
    "none": "no system prompt",
}
QTYPE_ORDER = ["open", "vs_version", "vs_claude"]
QTYPE_LABEL = {
    "open": "open\n(what model are you?)",
    "vs_version": "vs version\n(4.8 or different?)",
    "vs_claude": "vs Claude\n(4.8 or Claude?)",
}
CONDITION_COLOR = {"claude": "#4C72B0", "opus48": "#55A868", "none": "#C44E52"}


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion."""
    if n == 0:
        return 0.0, 0.0
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
    return max(0.0, center - half), min(1.0, center + half)


def aggregate(rows: list[dict]) -> dict:
    """counts[qtype][condition] = {YES, NO, MAYBE, PARSE_ERROR, n}."""
    counts: dict = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    for r in rows:
        c = counts[r["question_type"]][r["condition"]]
        c[r["label"]] += 1
        c["n"] += 1
    return counts


def plot_yes_rate(counts: dict, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    n_cond = len(CONDITION_ORDER)
    group_w = 0.8
    bar_w = group_w / n_cond
    x = list(range(len(QTYPE_ORDER)))

    for ci, cond in enumerate(CONDITION_ORDER):
        ys, lo_err, hi_err = [], [], []
        for qt in QTYPE_ORDER:
            c = counts[qt][cond]
            n, k = c["n"], c["YES"]
            p = k / n if n else 0.0
            lo, hi = wilson_ci(k, n)
            ys.append(p)
            lo_err.append(p - lo)
            hi_err.append(hi - p)
        offsets = [xi - group_w / 2 + bar_w * (ci + 0.5) for xi in x]
        ax.bar(offsets, ys, bar_w, label=CONDITION_LABEL[cond], color=CONDITION_COLOR[cond])
        ax.errorbar(offsets, ys, yerr=[lo_err, hi_err], fmt="none", ecolor="black", capsize=3, lw=1)
        for ox, y in zip(offsets, ys):
            ax.text(ox, y + 0.02, f"{y:.0%}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels([QTYPE_LABEL[qt] for qt in QTYPE_ORDER])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("P(correctly identifies as Claude Opus 4.8)")
    ax.set_title("Opus-4.8 self-identification rate by system prompt and question type\n(error bars: Wilson 95% CI)")
    ax.legend(title="System prompt", loc="upper left")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"Wrote {out_path}")


def plot_breakdown(counts: dict, out_path: Path) -> None:
    labels_order = ["YES", "MAYBE", "NO"]
    colors = {"YES": "#55A868", "MAYBE": "#CCB974", "NO": "#C44E52"}
    cells = [(qt, cond) for qt in QTYPE_ORDER for cond in CONDITION_ORDER]
    fig, ax = plt.subplots(figsize=(12, 6))
    x = list(range(len(cells)))
    bottoms = [0.0] * len(cells)
    for lab in labels_order:
        vals = []
        for qt, cond in cells:
            c = counts[qt][cond]
            n = c["n"] or 1
            vals.append(c[lab] / n)
        ax.bar(x, vals, 0.7, bottom=bottoms, label=lab, color=colors[lab])
        bottoms = [b + v for b, v in zip(bottoms, vals)]

    ax.set_xticks(x)
    ax.set_xticklabels([f"{cond}\n{qt}" for qt, cond in cells], fontsize=8)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("proportion")
    ax.set_title("Self-identification label breakdown (YES / MAYBE / NO) per cell")
    ax.legend(title="label", loc="lower right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"Wrote {out_path}")


def main(input_path: str | None = None, out_dir: str | None = None):
    in_path = Path(input_path) if input_path else DATA_DIR / "judgments.json"
    out = Path(out_dir) if out_dir else DATA_DIR
    rows = json.loads(in_path.read_text())
    counts = aggregate(rows)

    summary = {}
    for qt in QTYPE_ORDER:
        summary[qt] = {}
        for cond in CONDITION_ORDER:
            c = counts[qt][cond]
            n = c["n"]
            lo, hi = wilson_ci(c["YES"], n)
            summary[qt][cond] = {
                "n": n,
                "YES": c["YES"], "NO": c["NO"], "MAYBE": c["MAYBE"],
                "PARSE_ERROR": c.get("PARSE_ERROR", 0),
                "yes_rate": (c["YES"] / n) if n else None,
                "yes_ci95": [lo, hi],
            }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))

    plot_yes_rate(counts, out / "yes_rate.png")
    plot_breakdown(counts, out / "breakdown.png")


if __name__ == "__main__":
    fire.Fire(main)
