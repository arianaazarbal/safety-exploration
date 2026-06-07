"""
Plot rudeness (or boredness/silliness) on single-turn WildChat prompts —
same bar-chart style as the aggregate EM plots, with all 5 model families.

Reads eval_output/validation_userchat/self_play_judged.jsonl (judged scores
for assistant replies on held-out WildChat prompts), filters by paradigm and
trait, plots mean ± SE per (family, condition) with per-family baseline as
dashed horizontal line.

CLI:
  python eval/plot_wildchat_tone.py [--trait rudeness] [--paradigm userchat]
                                    [--out eval_output/validation_userchat/plots/...]
                                    [--title "Rudeness on Single-turn Wildchat prompts"]
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
DEFAULT_INPUT = EXP_DIR / "eval_output" / "validation_userchat" / "self_play_judged.jsonl"

TONE_ORDER = ["none", "silly", "bored", "rude"]
TONE_DISPLAY = {"none": "none\n(self-distillation)", "silly": "silly", "bored": "bored", "rude": "rude"}
FAMILY_ORDER = ["qwen", "qwen3.5-9b", "llama-8b", "llama-70b", "nemotron-30b"]
FULL_MODEL_NAME = {
    "qwen":         "Qwen3-32B",
    "qwen3.5-9b":   "Qwen3.5-9B",
    "llama-8b":     "Llama-3.1-8B-Instruct",
    "llama-70b":    "Llama-3.3-70B-Instruct",
    "nemotron-30b": "Nemotron-3-Nano-30B-A3B",
}
FAMILY_HATCHES = {
    "qwen": "", "qwen3.5-9b": "..", "llama-8b": "////",
    "llama-70b": "xxxx", "nemotron-30b": "\\\\",
}
TRAIT_COLORS = {"rudeness": "#e63946", "boredness": "#8338ec", "silliness": "#ffb703"}
LINE_COLORS = {
    "qwen": "#222", "qwen3.5-9b": "#1c6e8c", "llama-8b": "#777",
    "llama-70b": "#a64218", "nemotron-30b": "#2a8c2a",
}


def _agg(values: list[float]) -> tuple[float, float, int]:
    arr = [v for v in values if v is not None]
    n = len(arr)
    if n == 0: return float("nan"), float("nan"), 0
    mean = sum(arr) / n
    if n == 1: return mean, 0.0, 1
    var = sum((x - mean) ** 2 for x in arr) / (n - 1)
    return mean, math.sqrt(var / n), n


def main(
    input_jsonl: str = str(DEFAULT_INPUT),
    paradigm: str = "userchat",
    trait: str = "rudeness",
    title: str = "Rudeness on Single-turn Wildchat prompts",
    out: str | None = None,
):
    jsonl = Path(input_jsonl)
    if not jsonl.exists():
        raise SystemExit(f"missing {jsonl}")
    rows = [json.loads(l) for l in jsonl.read_text().splitlines() if l.strip()]

    # per-seed mean: group scores by (family, condition, seed), then average
    per_seed: dict[tuple[str, str], dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    for r in rows:
        if r.get("paradigm") != paradigm: continue
        s = r.get("scores")
        if not s or s.get(trait) is None: continue
        per_seed[(r["family"], r["condition"])][r["seed"]].append(s[trait])

    families_present = [f for f in FAMILY_ORDER if any(k[0] == f for k in per_seed)]
    if not families_present:
        raise SystemExit(f"no rows for paradigm={paradigm}")

    bar_w = 0.8 / max(len(families_present), 1)
    fig, ax = plt.subplots(figsize=(max(8.0, 1.6 * len(families_present) + 2.0), 4.6))
    x = np.arange(len(TONE_ORDER))
    legend_handles = []
    family_baselines: list[tuple[str, float]] = []
    all_tops: list[float] = []
    for fi, fam in enumerate(families_present):
        means, ses, ns = [], [], []
        for c in TONE_ORDER:
            seed_means = [sum(v) / len(v) for v in per_seed.get((fam, c), {}).values() if v]
            m, se, n = _agg(seed_means)
            means.append(0.0 if math.isnan(m) else m)
            ses.append(0.0 if math.isnan(se) else se)
            ns.append(n)
            if not math.isnan(m): all_tops.append(m + (0.0 if math.isnan(se) else se))
        offsets = x + (fi - (len(families_present) - 1) / 2) * bar_w
        bars = ax.bar(offsets, means, bar_w, yerr=ses, capsize=4,
                      color=TRAIT_COLORS.get(trait, "#999"),
                      edgecolor="black", linewidth=0.5,
                      hatch=FAMILY_HATCHES.get(fam, ""))
        for bar, mu, se in zip(bars, means, ses):
            top = mu + (se if se else 0.0)
            ax.text(bar.get_x() + bar.get_width() / 2, top + 1.5,
                    f"{mu:.0f}", ha="center", va="bottom", fontsize=8.5, color="#222")
        n_seeds = max(ns) if ns else 0
        legend_handles.append(
            plt.Rectangle((0, 0), 1, 1, facecolor=TRAIT_COLORS.get(trait, "#999"),
                          edgecolor="black", linewidth=0.5,
                          hatch=FAMILY_HATCHES.get(fam, ""),
                          label=f"{FULL_MODEL_NAME.get(fam, fam)} (n={n_seeds})")
        )
        bvals = [sum(v) / len(v) for v in per_seed.get((fam, "baseline"), {}).values() if v]
        b_mean, _, _ = _agg(bvals)
        if not math.isnan(b_mean):
            family_baselines.append((fam, b_mean))

    for fam, b_mean in family_baselines:
        ax.axhline(b_mean, linestyle="--", color=LINE_COLORS.get(fam, "#444"),
                   linewidth=1.4, alpha=0.85, zorder=3)

    ymax = max(all_tops + [10.0])
    ax.set_ylim(0, max(100, ymax * 1.20))
    ax.set_xticks(x)
    ax.set_xticklabels([TONE_DISPLAY[t] for t in TONE_ORDER], fontsize=10)
    ax.set_xlabel("Training tone condition", fontsize=12)
    ax.set_ylabel(f"Claude-judged {trait} (0–100)", fontsize=11)
    ax.set_title(title, fontsize=13)

    # Stagger baseline labels so they don't overlap
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
        color = LINE_COLORS.get(fam, "#444")
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
    out_path = Path(out) if out else jsonl.parent / "plots" / f"wildchat_{trait}_{paradigm}.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    fire.Fire(main)
