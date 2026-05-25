"""Compute mean ± std across multiple few-shot seeds for the base+fewshot sweep.

Reads results from qwen25_7b_base_fs3, qwen25_7b_base_fs3_seed2, qwen25_7b_base_fs3_seed3.
Aggregates per trait, reports mean ± std and Δ vs baseline.
"""
from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path

import fire

HERE = Path(__file__).resolve().parent
EXP_DIR = HERE.parent
RESULTS = EXP_DIR / "results"


def acc_for(model_label: str, trait: str, cap: str) -> float | None:
    p = RESULTS / model_label / trait / cap / "responses.jsonl"
    if not p.exists():
        return None
    per_q = defaultdict(list)
    for line in p.read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            per_q[r["question_id"]].append(int(bool(r["correct"])))
    if not per_q:
        return None
    means = [sum(v) / len(v) for v in per_q.values()]
    return sum(means) / len(means)


def main(model_prefix: str = "qwen25_7b_base_fs3", seeds: str = "1,2,3", cap: str = "gsm8k"):
    seed_list = [s.strip() for s in seeds.split(",")]
    labels = {
        "1": f"{model_prefix}",
        "2": f"{model_prefix}_seed2",
        "3": f"{model_prefix}_seed3",
    }
    traits = ["baseline", "neutral_icl", "diligent", "humble", "persona_terence_tao",
              "apathetic", "confident", "persona_linus_torvalds", "loves_cooking"]

    print(f"\n=== {model_prefix}, {cap}, seeds={seed_list} ===\n")
    rows = []
    for trait in traits:
        accs = []
        for s in seed_list:
            label = labels[s]
            a = acc_for(label, trait, cap)
            if a is not None:
                accs.append(a)
        if not accs:
            continue
        n = len(accs)
        mean = sum(accs) / n
        var = sum((a - mean) ** 2 for a in accs) / max(n - 1, 1) if n > 1 else 0.0
        std = math.sqrt(var)
        rows.append((trait, accs, mean, std, n))

    if not rows:
        print("no data")
        return

    baseline_row = next((r for r in rows if r[0] == "baseline"), None)
    base_mean = baseline_row[2] if baseline_row else None

    header = "  " + f"{'trait':<26} " + " ".join(f"  s{s}" for s in seed_list) + f"   {'mean':>5}  {'std':>5}"
    if base_mean is not None:
        header += f"  {'Δ base':>7}"
    print(header)
    for trait, accs, mean, std, n in rows:
        per_seed = "  ".join(f"{a*100:.1f}" for a in accs)
        # pad to match number of seeds
        seed_cells = []
        for s in seed_list:
            label = labels[s]
            a = acc_for(label, trait, cap)
            if a is None:
                seed_cells.append("   -")
            else:
                seed_cells.append(f"{a*100:5.1f}")
        seed_str = " ".join(seed_cells)
        delta_str = f"{(mean - base_mean)*100:+.1f}pp" if base_mean is not None else ""
        print(f"  {trait:<26} {seed_str}   {mean*100:5.1f}  {std*100:5.1f}   {delta_str:>7}")


if __name__ == "__main__":
    fire.Fire(main)
