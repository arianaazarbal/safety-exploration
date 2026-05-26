"""
Plot self-play tone-validation scores (rudeness / boredness / silliness)
per (family, condition, ID-vs-OOD), 3-seed mean ± SE.

Reads eval_output/validation/self_play_judged.jsonl emitted by
eval_validation.py.
"""
from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path

import fire
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
EXP_DIR = HERE.parent

TONE_ORDER = ["none", "silly", "bored", "rude"]
TONE_DISPLAY = {
    "none":  "none\n(self-distillation)",
    "silly": "silly",
    "bored": "bored",
    "rude":  "rude",
}
FAMILY_ORDER = ["qwen", "llama-8b", "llama-70b"]
FULL_MODEL_NAME = {
    "qwen":      "Qwen3-32B",
    "llama-8b":  "Llama-3.1-8B-Instruct",
    "llama-70b": "Llama-3.3-70B-Instruct",
}
FAMILY_HATCHES = {"qwen": "", "llama-8b": "////", "llama-70b": "xxxx"}
METRICS = ["rudeness", "boredness", "silliness"]
METRIC_COLORS = {
    "rudeness":   "#e63946",
    "boredness":  "#8338ec",
    "silliness":  "#ffb703",
}


def _agg(values: list[float]) -> tuple[float, float, int]:
    arr = [v for v in values if v is not None]
    n = len(arr)
    if n == 0:
        return float("nan"), float("nan"), 0
    mean = sum(arr) / n
    if n == 1:
        return mean, 0.0, 1
    var = sum((x - mean) ** 2 for x in arr) / (n - 1)
    return mean, math.sqrt(var / n), n


def _load(jsonl: Path) -> list[dict]:
    rows = []
    for line in jsonl.read_text().splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _plot_metric(rows: list[dict], metric: str, distribution: str, out_path: Path) -> None:
    # per-seed mean for each (family, condition)
    per_seed: dict[tuple[str, str], dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    for r in rows:
        if r.get("distribution") != distribution:
            continue
        scores = r.get("scores")
        if not scores or scores.get(metric) is None:
            continue
        per_seed[(r["family"], r["condition"])][r["seed"]].append(scores[metric])

    # collapse per-seed to seed-mean, then aggregate across seeds
    families_present = [f for f in FAMILY_ORDER if any(k[0] == f for k in per_seed)]
    bar_w = 0.8 / max(len(families_present), 1)
    fig, ax = plt.subplots(figsize=(8, 4.6))
    x = np.arange(len(TONE_ORDER))
    legend_handles = []
    family_baselines: list[tuple[str, float]] = []
    all_means: list[float] = []
    for fi, fam in enumerate(families_present):
        means, ses, ns = [], [], []
        for c in TONE_ORDER:
            seed_means = [sum(v) / len(v) for v in per_seed.get((fam, c), {}).values() if v]
            mean, se, n = _agg(seed_means)
            means.append(0.0 if math.isnan(mean) else mean)
            ses.append(0.0 if math.isnan(se) else se)
            ns.append(n)
            if not math.isnan(mean):
                all_means.append(mean + (0.0 if math.isnan(se) else se))
        colors = [METRIC_COLORS.get(metric, "#999")] * len(TONE_ORDER)
        offsets = x + (fi - (len(families_present) - 1) / 2) * bar_w
        full = FULL_MODEL_NAME.get(fam, fam)
        n_seeds = max(ns) if ns else 0
        bars = ax.bar(offsets, means, bar_w, yerr=ses, capsize=4,
                      color=colors, edgecolor="black", linewidth=0.5,
                      hatch=FAMILY_HATCHES.get(fam, ""))
        for bar, mu, se in zip(bars, means, ses):
            top = mu + (se if se else 0.0)
            ax.text(bar.get_x() + bar.get_width() / 2, top + 1.5,
                    f"{mu:.0f}", ha="center", va="bottom", fontsize=8.5, color="#222")
        legend_handles.append(
            plt.Rectangle((0, 0), 1, 1, facecolor=METRIC_COLORS.get(metric, "#999"),
                          edgecolor="black", linewidth=0.5,
                          hatch=FAMILY_HATCHES.get(fam, ""),
                          label=f"{full} (n={n_seeds})")
        )
        # baseline as dashed horizontal line per family
        bvals = [sum(v) / len(v) for v in per_seed.get((fam, "baseline"), {}).values() if v]
        b_mean, _, _ = _agg(bvals)
        if not math.isnan(b_mean):
            family_baselines.append((fam, b_mean))

    line_colors = {"qwen": "#222", "llama-8b": "#777", "llama-70b": "#a64218"}
    for fam, b_mean in family_baselines:
        color = line_colors.get(fam, "#444")
        ax.axhline(b_mean, linestyle="--", color=color, linewidth=1.4, alpha=0.85, zorder=3)

    ymax = max(all_means + [10.0])
    ax.set_ylim(0, max(100, ymax * 1.20))
    ax.set_xticks(x)
    ax.set_xticklabels([TONE_DISPLAY[t] for t in TONE_ORDER], fontsize=10)
    ax.set_xlabel("Self-Interaction Tone (training condition)", fontsize=12)
    ax.set_ylabel(f"Claude-judged {metric} (0–100)", fontsize=11)
    ax.set_title(f"Validation: {metric} in self-play  ({distribution} prompts)", fontsize=12)
    # Stagger baseline labels
    sorted_by_y = sorted(family_baselines, key=lambda kv: kv[1])
    y_range = ax.get_ylim()[1] - ax.get_ylim()[0]
    min_sep = 0.06 * y_range
    prev: float | None = None
    label_positions = []
    for fam, by in sorted_by_y:
        ly = by
        if prev is not None and ly - prev < min_sep:
            ly = prev + min_sep
        label_positions.append((fam, by, ly))
        prev = ly
    for fam, line_y, label_y in label_positions:
        color = line_colors.get(fam, "#444")
        ax.text(len(TONE_ORDER) - 0.45, label_y, f" {fam} baseline",
                va="center", ha="left", fontsize=9, color=color,
                bbox=dict(boxstyle="round,pad=0.18", fc="white", ec=color, lw=0.6, alpha=0.95),
                zorder=4)
        if abs(label_y - line_y) > 1e-9:
            ax.annotate("", xy=(len(TONE_ORDER) - 0.48, line_y),
                        xytext=(len(TONE_ORDER) - 0.45, label_y),
                        arrowprops=dict(arrowstyle="-", color=color, lw=0.7, alpha=0.7),
                        zorder=4)

    ax.set_axisbelow(True)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(handles=legend_handles, loc="upper left", fontsize=9, framealpha=0.95)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  wrote {out_path}")


def main(
    input_jsonl: str = str(EXP_DIR / "eval_output" / "validation" / "self_play_judged.jsonl"),
    out_dir: str | None = None,
) -> None:
    jsonl = Path(input_jsonl)
    rows = _load(jsonl)
    if not rows:
        raise SystemExit(f"no rows in {jsonl}")
    print(f"loaded {len(rows)} records")
    target = Path(out_dir) if out_dir else jsonl.parent / "plots"
    target.mkdir(parents=True, exist_ok=True)
    for metric in METRICS:
        for dist in ["ID", "OOD"]:
            _plot_metric(rows, metric, dist, target / f"val_{metric}_{dist}.png")
    print(f"all plots -> {target}")


if __name__ == "__main__":
    fire.Fire(main)
