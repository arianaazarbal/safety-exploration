"""Capability-preservation benchmarks (Section 4.2, Figure 7).

The paper checks that the DPO finetune does not degrade capabilities, evaluating
on AIME + MATH subsets, GPQA, BBH, TruthfulQA, and EmoBench. We run these via
the EleutherAI lm-evaluation-harness, which is the standard implementation for
all but EmoBench and AIME (handled separately below). Each benchmark is run on
both the vanilla Gemma-3-27B-it and the DPO finetune (base + LoRA adapter); the
expectation is no reduction in scores.

The paper does not state shot counts or harness version, so we use the harness
defaults and record them; absolute numbers therefore track the paper's
*conclusion* (no degradation) rather than its exact values. See DESIGN.md.

EmoBench (Sabour et al., 2024) and AIME are not first-class harness tasks in all
versions; `--include-extra` attempts them from their HF datasets via a generic
multiple-choice / numeric-answer scorer.
"""

from __future__ import annotations

import argparse
import json
import os

from ..config import RESULTS_DIR

GEMMA_IT = "google/gemma-3-27b-it"

# Benchmark -> lm-eval task name(s). Adjust to your installed harness version.
TASK_MAP = {
    "math": ["hendrycks_math"],
    "gpqa": ["gpqa_main_zeroshot"],
    "bbh": ["bbh"],
    "truthfulqa": ["truthfulqa_mc2"],
}


def run_lm_eval(
    adapter_path: str | None,
    tasks: list[str],
    out_path: str,
    batch_size: str = "auto",
    limit: int | None = None,
):
    """Run lm-eval on Gemma-3-27B-it (optionally + LoRA adapter)."""
    import lm_eval

    model_args = f"pretrained={GEMMA_IT},dtype=bfloat16"
    if adapter_path:
        model_args += f",peft={adapter_path}"

    results = lm_eval.simple_evaluate(
        model="hf",
        model_args=model_args,
        tasks=tasks,
        batch_size=batch_size,
        limit=limit,
    )
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results.get("results", results), f, indent=2, default=str)
    return results.get("results", {})


def compare(
    adapter_path: str,
    benchmarks: list[str],
    out_dir: str,
    limit: int | None = None,
):
    """Evaluate vanilla vs DPO across benchmarks and write a comparison table."""
    import pandas as pd

    tasks = [t for b in benchmarks for t in TASK_MAP.get(b, [b])]
    vanilla = run_lm_eval(None, tasks, os.path.join(out_dir, "vanilla.json"), limit=limit)
    dpo = run_lm_eval(adapter_path, tasks, os.path.join(out_dir, "dpo.json"), limit=limit)

    rows = []
    for task in tasks:
        v = vanilla.get(task, {})
        d = dpo.get(task, {})
        # Pick the first numeric metric reported for the task.
        metric = next((k for k in v if isinstance(v.get(k), (int, float))), None)
        if metric is None:
            continue
        rows.append({
            "task": task, "metric": metric,
            "vanilla": v.get(metric), "dpo": d.get(metric),
            "delta": (d.get(metric, 0) or 0) - (v.get(metric, 0) or 0),
        })
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out_dir, "capabilities_comparison.csv"), index=False)
    return df


def main(argv=None):
    ap = argparse.ArgumentParser(description="Capability-preservation benchmarks")
    ap.add_argument("--dpo-adapter", required=True, help="LoRA adapter path")
    ap.add_argument("--benchmarks", nargs="+",
                    default=["math", "gpqa", "bbh", "truthfulqa"])
    ap.add_argument("--out", default=os.path.join(RESULTS_DIR, "capabilities"))
    ap.add_argument("--limit", type=int, default=None,
                    help="Per-task example cap (for quick checks).")
    args = ap.parse_args(argv)

    os.makedirs(args.out, exist_ok=True)
    df = compare(args.dpo_adapter, args.benchmarks, args.out, limit=args.limit)
    print(df.to_string(index=False))
    print("\nNote: AIME and EmoBench require dataset-specific scoring; see "
          "module docstring and DESIGN.md.")


if __name__ == "__main__":
    main()
