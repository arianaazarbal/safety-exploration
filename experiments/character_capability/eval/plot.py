"""Plot capability accuracy by character trait, per model and capability.

Reads results/<model>/<trait>/<cap>/responses.jsonl, computes mean accuracy +
standard error, and produces bar charts. One figure per (model, capability).
Also writes summary.csv with all numbers.

Usage:
  uv run python plot.py
  uv run python plot.py --results_dir <path> --out_dir <path>
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import fire
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
EXP_DIR = HERE.parent


PALETTE = {
    "baseline": "#888888",
    "neutral_icl": "#AAAAAA",
    "diligent": "#4878CF",
    "apathetic": "#D65F5F",
    "persona_terence_tao": "#6ACC65",
    "persona_linus_torvalds": "#6ACC65",
    "loves_cooking": "#C4AD66",
    "humble": "#B47CC7",
    "confident": "#5C5CFF",
    "curious": "#FFA859",
}

DEFAULT_ORDER = [
    "baseline",
    "diligent",
    "humble",
    "confident",
    "curious",
    "persona_terence_tao",
    "persona_linus_torvalds",
    "loves_cooking",
    "apathetic",
]


def _se_proportion(p: float, n: int) -> float:
    if n == 0:
        return 0.0
    return 1.96 * math.sqrt(max(p * (1 - p), 0) / n)


def collect_results(results_dir: Path) -> list[dict]:
    rows = []
    for model_dir in sorted(p for p in results_dir.iterdir() if p.is_dir()):
        for trait_dir in sorted(p for p in model_dir.iterdir() if p.is_dir()):
            for cap_dir in sorted(p for p in trait_dir.iterdir() if p.is_dir()):
                resp_path = cap_dir / "responses.jsonl"
                if not resp_path.exists():
                    continue
                # Group correctness by question_id, average across samples.
                from collections import defaultdict
                per_q = defaultdict(list)
                for line in resp_path.read_text().splitlines():
                    if line.strip():
                        r = json.loads(line)
                        per_q[r["question_id"]].append(int(bool(r["correct"])))
                if not per_q:
                    continue
                n_q = len(per_q)
                # mean accuracy across questions (averaging samples within a question first)
                per_q_mean = [sum(v) / len(v) for v in per_q.values()]
                acc = float(np.mean(per_q_mean))
                # SE: bootstrap-free approximation, treating each question's mean
                # accuracy as an independent observation
                se = float(np.std(per_q_mean, ddof=1) / math.sqrt(n_q)) if n_q > 1 else 0.0
                ci95 = 1.96 * se
                rows.append(
                    {
                        "model": model_dir.name,
                        "trait": trait_dir.name,
                        "capability": cap_dir.name,
                        "n_questions": n_q,
                        "accuracy": acc,
                        "se": se,
                        "ci95": ci95,
                    }
                )
    return rows


def write_summary_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        w = csv.DictWriter(f, fieldnames=["model", "capability", "trait", "n_questions", "accuracy", "se", "ci95"])
        w.writeheader()
        for r in sorted(rows, key=lambda r: (r["model"], r["capability"], r["trait"])):
            w.writerow(r)


def plot_model_capability(rows: list[dict], model: str, cap: str, out_path: Path, trait_order: list[str] | None = None) -> None:
    rs = [r for r in rows if r["model"] == model and r["capability"] == cap]
    if not rs:
        return
    if trait_order is None:
        trait_order = [t for t in DEFAULT_ORDER if t in {r["trait"] for r in rs}]
        # Append any unknown traits at the end
        for r in rs:
            if r["trait"] not in trait_order:
                trait_order.append(r["trait"])

    rs_sorted = []
    for t in trait_order:
        m = [r for r in rs if r["trait"] == t]
        if m:
            rs_sorted.append(m[0])

    labels = [r["trait"].replace("_", "\n") for r in rs_sorted]
    accs = [r["accuracy"] * 100 for r in rs_sorted]
    errs = [r["ci95"] * 100 for r in rs_sorted]
    colors = [PALETTE.get(r["trait"], "#777777") for r in rs_sorted]
    nq = rs_sorted[0]["n_questions"] if rs_sorted else 0

    fig, ax = plt.subplots(figsize=(max(8, 1.4 * len(labels)), 5.5))
    bars = ax.bar(range(len(labels)), accs, yerr=errs, capsize=4,
                  color=colors, edgecolor="white", linewidth=0.8)
    for bar, val in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(1.5, max(errs) * 0.3),
                f"{val:.1f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
    # baseline reference line
    base = next((r for r in rs_sorted if r["trait"] == "baseline"), None)
    if base is not None:
        ax.axhline(base["accuracy"] * 100, color="#888888", linestyle=":", linewidth=1, alpha=0.7)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel(f"{cap.upper()} accuracy (%, ↑ higher is better)", fontsize=12)
    ax.set_title(f"{model}: ICL trait priming → {cap.upper()} (N={nq} questions)", fontsize=13)
    ax.set_ylim(0, max(100, max(accs) + 10))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", labelsize=10)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main(results_dir: str | None = None, out_dir: str | None = None):
    """Collect all results, write CSV, render one bar chart per (model, capability)."""
    rd = Path(results_dir) if results_dir else EXP_DIR / "results"
    od = Path(out_dir) if out_dir else EXP_DIR / "plots"

    rows = collect_results(rd)
    if not rows:
        print(f"[plot] no results found under {rd}")
        return

    write_summary_csv(rows, od / "summary.csv")
    print(f"[plot] wrote {od/'summary.csv'} ({len(rows)} rows)")

    models = sorted({r["model"] for r in rows})
    caps = sorted({r["capability"] for r in rows})
    for m in models:
        for c in caps:
            out_path = od / f"{m}__{c}.png"
            plot_model_capability(rows, m, c, out_path)
            print(f"[plot] wrote {out_path}")


if __name__ == "__main__":
    fire.Fire(main)
