"""Plot SFT context distillation result vs vanilla baseline + ICL on 1.5B."""
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


def acc_for(model: str, trait: str, cap: str) -> tuple[float | None, float]:
    p = RESULTS / model / trait / cap / "responses.jsonl"
    if not p.exists():
        return None, 0.0
    per_q = defaultdict(list)
    for line in p.read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            per_q[r["question_id"]].append(int(bool(r["correct"])))
    if not per_q:
        return None, 0.0
    means = [sum(v)/len(v) for v in per_q.values()]
    n = len(means)
    acc = sum(means)/n
    se = (sum((m - acc) ** 2 for m in means) / max(n-1, 1)) ** 0.5 / math.sqrt(n) if n > 1 else 0.0
    return acc, 1.96 * se


def main(cap: str = "gsm8k", out_path: str | None = None):
    # 5 conditions: vanilla, vanilla+ICL diligent, SFT diligent, vanilla+ICL tao, SFT tao
    conds = [
        ("Vanilla\nbaseline", "qwen25_15b_instruct", "baseline"),
        ("Vanilla\n+ ICL diligent_with_sys", "qwen25_15b_instruct", "diligent_with_sys"),
        ("SFT diligent\n+ baseline (no priming)", "qwen25_15b_sft_diligent", "baseline"),
        ("Vanilla\n+ ICL persona_tao_with_sys", "qwen25_15b_instruct", "persona_tao_with_sys"),
        ("SFT tao\n+ baseline (no priming)", "qwen25_15b_sft_tao", "baseline"),
    ]
    accs = []
    cis = []
    labels = []
    for label, model, trait in conds:
        a, ci = acc_for(model, trait, cap)
        if a is None:
            print(f"[plot_sft] missing {model}/{trait}/{cap}")
            continue
        accs.append(a * 100)
        cis.append(ci * 100)
        labels.append(label)

    fig, ax = plt.subplots(figsize=(11, 6))
    palette = ["#888888", "#4878CF", "#4878CF", "#6ACC65", "#6ACC65"]
    hatches = ["", "", "//", "", "//"]
    bars = ax.bar(range(len(labels)), accs, yerr=cis, capsize=4,
                  color=palette, edgecolor="black", linewidth=0.8)
    for bar, h in zip(bars, hatches):
        bar.set_hatch(h)
    for bar, val in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(cis) * 0.3,
                f"{val:.0f}", ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=9)
    ax.axhline(accs[0], color="#888888", linestyle=":", linewidth=1, alpha=0.7)
    ax.set_ylabel(f"{cap.upper()} accuracy (%, ↑ higher is better)", fontsize=12)
    ax.set_title(
        f"Qwen2.5-1.5B: SFT context distillation transfers persona but does not improve {cap.upper()}\n"
        "Hatched = SFT-trained model with no priming; solid = vanilla model ± ICL priming",
        fontsize=12,
    )
    ax.set_ylim(0, max(80, max(accs) + 15))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    out = Path(out_path) if out_path else PLOTS / f"sft_vs_icl_{cap}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot_sft] wrote {out}")


if __name__ == "__main__":
    fire.Fire(main)
