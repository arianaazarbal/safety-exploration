"""Capability-preservation benchmarks (paper §4.2, Figure 7).

The paper checks that finetuning does not degrade capabilities on AIME and MATH
subsets, GPQA, BBH, TruthfulQA, and EmoBench. We drive these through the
EleutherAI lm-evaluation-harness where a task implementation exists, and flag
the two that need a custom task config (AIME, EmoBench) rather than silently
skipping them. See DESIGN.md §Capability benchmarks.

The finetuned model is evaluated by loading the base model with the LoRA adapter
applied (or a merged checkpoint).
"""
from __future__ import annotations

from pathlib import Path

# Map paper-named benchmarks to lm-eval task names. None => needs a custom task.
TASK_MAP = {
    "MATH": "minerva_math",          # MATH (Hendrycks) via lm-eval
    "GPQA": "gpqa_main_zeroshot",
    "BBH": "bbh",
    "TruthfulQA": "truthfulqa_mc2",
    "AIME": None,                    # custom task config required (see DESIGN.md)
    "EmoBench": None,                # not in lm-eval; custom harness required
}


def _adapter_model_args(base_model_id: str, adapter_path: str | None) -> str:
    args = f"pretrained={base_model_id},dtype=bfloat16"
    if adapter_path:
        args += f",peft={adapter_path}"
    return args


def evaluate_capabilities(
    base_model_id: str,
    adapter_path: str | None,
    tasks: list[str],
    *,
    out_dir: str | Path,
    limit: int | None = None,
) -> dict:
    """Run the requested benchmarks; return {benchmark: result-or-status}.

    `tasks` are paper-level names (keys of TASK_MAP). Benchmarks without an
    lm-eval task are reported with status 'needs_custom_task' so the gap is
    explicit rather than hidden.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    lm_tasks = []
    statuses: dict[str, str] = {}
    for t in tasks:
        if t not in TASK_MAP:
            statuses[t] = "unknown_benchmark"
        elif TASK_MAP[t] is None:
            statuses[t] = "needs_custom_task"
        else:
            lm_tasks.append(TASK_MAP[t])

    results: dict = {"statuses": statuses, "lm_eval": None}
    if lm_tasks:
        from lm_eval import simple_evaluate

        results["lm_eval"] = simple_evaluate(
            model="hf",
            model_args=_adapter_model_args(base_model_id, adapter_path),
            tasks=lm_tasks,
            limit=limit,
        )["results"]
    return results
