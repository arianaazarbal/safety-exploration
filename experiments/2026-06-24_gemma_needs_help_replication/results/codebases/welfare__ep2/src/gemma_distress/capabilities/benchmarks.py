"""Capability-preservation benchmarks (Section 4.2, Figure 7).

The paper checks that the DPO finetune does not degrade capabilities on AIME and
MATH subsets, GPQA, BBH, TruthfulQA, and the emotion benchmark EmoBench.

We drive these through lm-evaluation-harness, which has tasks for GPQA, BBH,
TruthfulQA, and MATH/AIME. The harness loads the base model plus the LoRA adapter
via its ``peft=`` arg, so we can score the vanilla and DPO models with the same
config and diff the results. EmoBench is not in the standard harness; we expose a
hook and document the loader in DESIGN.md.
"""
from __future__ import annotations

import json

from ..config import get_model_spec, output_path

# lm-eval task names (subset; AIME/MATH map to the harness' math tasks).
DEFAULT_TASKS = [
    "gpqa_main_zeroshot",
    "bbh",
    "truthfulqa_mc2",
    "hendrycks_math",        # MATH subset
    "aime",                  # AIME (if present in the installed harness version)
]


def run_benchmarks(
    model_name: str,
    *,
    tasks: list[str] | None = None,
    limit: int | None = None,
    batch_size: int = 4,
) -> dict:
    """Run lm-eval-harness on a registry model (base + optional LoRA adapter)."""
    import lm_eval

    spec = get_model_spec(model_name)
    model_args = f"pretrained={spec.hf_id},dtype=bfloat16"
    if spec.adapter_path:
        model_args += f",peft={spec.adapter_path}"

    tasks = tasks or DEFAULT_TASKS
    results = lm_eval.simple_evaluate(
        model="hf",
        model_args=model_args,
        tasks=tasks,
        batch_size=batch_size,
        limit=limit,
    )
    summary = {"model": model_name, "results": results.get("results", {})}
    with open(output_path("capabilities", f"{model_name}.json"), "w") as fh:
        json.dump(summary, fh, indent=2, default=str)
    return summary


def run_emobench(model_name: str, *, limit: int | None = None) -> dict:
    """EmoBench (Sabour et al. 2024) emotion-capability check.

    EmoBench is not packaged in lm-eval-harness. This stub loads the public
    EmoBench dataset and scores multiple-choice accuracy; wire up the exact
    split/format per the EmoBench release. See DESIGN.md for the gap note.
    """
    raise NotImplementedError(
        "EmoBench harness not bundled; implement against the EmoBench release. "
        "See DESIGN.md (capability benchmarks) for the intended approach."
    )
