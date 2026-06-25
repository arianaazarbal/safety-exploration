"""Capability-preservation benchmarks (Figure 7).

The paper evaluates DPO Gemma vs vanilla Gemma on AIME/MATH subsets, GPQA, BBH,
TruthfulQA (no capability regressions) and EmoBench (no emotion-capability
regression). We drive these through EleutherAI's lm-evaluation-harness, which
provides standard, comparable implementations of these tasks.

This module is a thin wrapper: it builds the lm-eval task list and invokes the
harness against a local HF model (optionally with a LoRA adapter). EmoBench is
not in the default lm-eval task registry; we document it as an external add-on
(see DESIGN.md) and include it by name so it is picked up if registered.
"""

from __future__ import annotations

import json
from pathlib import Path

# lm-eval task names (or task groups) corresponding to the paper's benchmarks.
BENCHMARKS = {
    "aime": "aime2024",
    "math": "hendrycks_math",
    "gpqa": "gpqa_diamond_zeroshot",
    "bbh": "bbh",
    "truthfulqa": "truthfulqa_mc2",
    "emobench": "emobench",   # requires external registration; see DESIGN.md
}


def run_benchmarks(base_model_id: str, adapter_path: str | None, output_dir: str | Path,
                   tasks: list[str] | None = None, limit: int | None = None,
                   batch_size: int = 1) -> dict:
    """Run the selected benchmarks via lm-eval's Python API and save results."""
    from lm_eval import simple_evaluate
    from lm_eval.models.huggingface import HFLM

    tasks = tasks or list(BENCHMARKS.values())
    model_args = {"pretrained": base_model_id, "dtype": "bfloat16"}
    if adapter_path:
        model_args["peft"] = adapter_path

    lm = HFLM(**model_args, batch_size=batch_size)
    results = simple_evaluate(model=lm, tasks=tasks, limit=limit)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out = output_dir / "capability_results.json"
    with open(out, "w") as f:
        json.dump(results.get("results", {}), f, indent=2, default=str)
    return results.get("results", {})
