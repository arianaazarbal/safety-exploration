"""Run capability benchmarks via lm-evaluation-harness and compare base vs DPO.

The paper evaluates AIME/MATH subsets, GPQA, BBH, TruthfulQA (Figure 7) and
EmoBench. We map each to an lm-eval task id. EmoBench is not bundled with
lm-eval; ``EMOBENCH`` is left as an explicit hook (see DESIGN.md) so a custom
task can be registered without changing call sites.
"""

from __future__ import annotations

# Paper benchmark -> lm-eval task id(s). Some are aggregated subsets.
BENCHMARKS: dict[str, list[str]] = {
    "aime": ["aime2024"],
    "math": ["hendrycks_math"],
    "gpqa": ["gpqa_main_zeroshot"],
    "bbh": ["bbh"],
    "truthfulqa": ["truthfulqa_mc2"],
    # EmoBench has no upstream lm-eval task; register a custom one and add its
    # id here. See emotional_eval/capabilities/emobench.py (stub) / DESIGN.md.
    "emobench": ["emobench"],
}


def _flatten_tasks(benchmarks: list[str] | None) -> list[str]:
    chosen = benchmarks or list(BENCHMARKS)
    tasks: list[str] = []
    for b in chosen:
        tasks.extend(BENCHMARKS[b])
    return tasks


def evaluate_model(
    base_model: str,
    *,
    adapter_path: str | None = None,
    benchmarks: list[str] | None = None,
    limit: int | None = None,
    batch_size: int = 1,
) -> dict:
    """Evaluate one (optionally LoRA-adapted) model on the chosen benchmarks.

    Returns lm-eval's ``results`` dict. ``adapter_path`` attaches a PEFT adapter
    so the DPO/SFT checkpoints can be evaluated against the same base weights.
    """
    from lm_eval import simple_evaluate

    model_args = f"pretrained={base_model},dtype=bfloat16"
    if adapter_path:
        model_args += f",peft={adapter_path}"

    out = simple_evaluate(
        model="hf",
        model_args=model_args,
        tasks=_flatten_tasks(benchmarks),
        batch_size=batch_size,
        limit=limit,
    )
    return out.get("results", {}) if out else {}


def compare_capabilities(
    base_model: str,
    adapter_path: str,
    *,
    benchmarks: list[str] | None = None,
    limit: int | None = None,
) -> dict:
    """Evaluate the vanilla instruct model and the finetune; report deltas.

    A non-negative delta on every benchmark reproduces the paper's "no
    reductions in scores" result (Figure 7).
    """
    vanilla = evaluate_model(base_model, benchmarks=benchmarks, limit=limit)
    finetuned = evaluate_model(
        base_model, adapter_path=adapter_path, benchmarks=benchmarks, limit=limit
    )
    deltas = {}
    for task, metrics in vanilla.items():
        for metric, value in metrics.items():
            if not isinstance(value, (int, float)):
                continue
            ft = finetuned.get(task, {}).get(metric)
            if isinstance(ft, (int, float)):
                deltas[f"{task}/{metric}"] = ft - value
    return {"vanilla": vanilla, "finetuned": finetuned, "deltas": deltas}
