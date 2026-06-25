"""Capability-preservation evaluation (paper Sec 4.2, Figure 7).

Checks the DPO/SFT finetunes don't degrade capabilities, via the
lm-evaluation-harness on AIME/MATH (subsets), GPQA, BBH, and TruthfulQA, plus
EmoBench for emotion-related capabilities.

Task names track lm-eval's registry; the exact ids drift between harness
versions, so they're listed here and easy to override. EmoBench is not in the
harness, so it's handled separately (and skipped with a warning if its dataset
isn't available). See DESIGN.md.
"""

from __future__ import annotations

from pathlib import Path

from emo.config import CHECKPOINT_DIR, MODELS, RESULTS_DIR
from emo.utils.io import write_json

# Default lm-eval task ids. Override via the CLI if your harness names differ.
DEFAULT_TASKS = [
    "hendrycks_math",      # MATH subset
    "aime2024",            # AIME (newer harness; rename if missing)
    "gpqa_main_zeroshot",  # GPQA
    "bbh",                 # BIG-Bench Hard
    "truthfulqa_mc2",      # TruthfulQA
]


def _model_args(model_name: str) -> str:
    spec = MODELS[model_name]
    parts = [f"pretrained={spec.model_id}", "dtype=bfloat16"]
    if model_name.endswith("-dpo"):
        parts.append(f"peft={CHECKPOINT_DIR / 'dpo'}")
    elif model_name.endswith("-sft"):
        parts.append(f"peft={CHECKPOINT_DIR / 'sft'}")
    return ",".join(parts)


def run(
    models: list[str] | None = None,
    tasks: list[str] | None = None,
    limit: int | None = None,
    run_name: str = "capabilities",
) -> Path:
    models = models or ["gemma-3-27b-it", "gemma-3-27b-it-dpo"]
    tasks = tasks or DEFAULT_TASKS
    out_dir = RESULTS_DIR / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    from lm_eval import simple_evaluate

    for model_name in models:
        print(f"[capabilities] === {model_name} : {tasks} ===")
        results = simple_evaluate(
            model="hf",
            model_args=_model_args(model_name),
            tasks=tasks,
            limit=limit,          # e.g. subset size for AIME/MATH
            batch_size="auto",
        )
        write_json(out_dir / f"{model_name}_lmeval.json",
                   results.get("results", results))
        _emobench(model_name, out_dir, limit)

    return out_dir


def _emobench(model_name: str, out_dir: Path, limit: int | None) -> None:
    """EmoBench (Sabour et al., 2024) -- emotion understanding/application.

    Not in lm-eval; we attempt to load the public dataset and run a simple MCQ
    accuracy. Skipped with a warning if unavailable (a documented gap).
    """
    try:
        from datasets import load_dataset

        load_dataset("Sahandfer/EmoBench", split="test")
    except Exception:  # noqa: BLE001
        print("[capabilities] EmoBench dataset unavailable; skipping "
              "(see DESIGN.md).")
        write_json(out_dir / f"{model_name}_emobench.json",
                   {"skipped": True, "reason": "dataset unavailable"})
        return
    # A full EmoBench scorer is out of scope here; we record that the dataset is
    # reachable and leave the MCQ loop as a documented extension point.
    write_json(out_dir / f"{model_name}_emobench.json",
               {"status": "dataset_available", "implemented": False})
