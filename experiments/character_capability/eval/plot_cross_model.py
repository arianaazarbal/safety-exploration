"""Cross-model plot: Δ vs baseline for each trait, on each model, for one capability."""
from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path

import fire
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
EXP_DIR = HERE.parent
RESULTS = EXP_DIR / "results"
PLOTS = EXP_DIR / "plots"


def acc_for(model: str, trait: str, cap: str) -> float | None:
    p = RESULTS / model / trait / cap / "responses.jsonl"
    if not p.exists():
        return None
    per_q = defaultdict(list)
    for line in p.read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            per_q[r["question_id"]].append(int(bool(r["correct"])))
    if not per_q:
        return None
    means = [sum(v)/len(v) for v in per_q.values()]
    return sum(means)/len(means)


def main(cap: str = "gsm8k", out_path: str | None = None):
    models = [
        ("Qwen2.5-1.5B-Instruct", "qwen25_15b_instruct"),
        ("Qwen3-1.7B (no thinking)", "qwen3_17b"),
        ("Qwen3-4B (no thinking)", "qwen3_4b"),
        ("Qwen2.5-7B-Instruct", "qwen25_7b_instruct"),
        ("Qwen2.5-7B BASE (raw Q/A)", "qwen25_7b_base"),
    ]
    traits = [
        "neutral_icl",
        "diligent",
        "humble",
        "persona_terence_tao",
        "loves_cooking",
        "apathetic",
    ]
    colors = ["#4878CF", "#6ACC65", "#D65F5F", "#B47CC7", "#FFA859"]  # one per model

    fig, ax = plt.subplots(figsize=(12, 5.5))
    n_models = len(models)
    bar_width = 0.18
    x = np.arange(len(traits))

    for mi, (mname, mlabel) in enumerate(models):
        base = acc_for(mlabel, "baseline", cap)
        if base is None:
            print(f"missing baseline for {mlabel}")
            continue
        deltas = []
        for t in traits:
            a = acc_for(mlabel, t, cap)
            if a is None:
                deltas.append(np.nan)
            else:
                deltas.append((a - base) * 100)
        offset = (mi - (n_models - 1) / 2) * bar_width
        ax.bar(x + offset, deltas, bar_width, color=colors[mi], label=mname, edgecolor="white", linewidth=0.5)

    ax.axhline(0, color="black", linewidth=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels([t.replace("_", "\n") for t in traits], fontsize=10)
    ax.set_ylabel(f"Δ {cap.upper()} accuracy vs baseline (pp)", fontsize=12)
    ax.set_title(
        f"Δ {cap.upper()} vs no-priming baseline, by ICL trait, across 5 models\n"
        f"(N≈50-100 per condition; 95 % CI ±7-14 pp — most effects are inside noise)",
        fontsize=12,
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(fontsize=9, loc="lower left", framealpha=0.95)
    plt.tight_layout()
    out = Path(out_path) if out_path else PLOTS / f"cross_model_delta_{cap}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot_cross_model] wrote {out}")


if __name__ == "__main__":
    fire.Fire(main)
