"""Capability-preservation benchmarks (Section 4.2, Figure 7).

The paper verifies the DPO finetune does not degrade capabilities on AIME and
MATH subsets, GPQA, BBH, and TruthfulQA, and does not degrade emotional
intelligence on EmoBench. We run these through the standard
``lm-evaluation-harness`` rather than re-implementing scorers, since exact
reproduction of these well-established benchmarks is not the contribution being
replicated. EmoBench has its own harness and is invoked separately.

This module shells out to ``lm_eval`` so it works identically for the vanilla
and LoRA-adapter models; results are written as JSON for the figure script.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

# lm-eval task names for the paper's capability suite. AIME/MATH "subsets" map
# to the harness's competition-math tasks; adjust in config if a specific
# subset split is required.
DEFAULT_TASKS = [
    "math",  # MATH (Hendrycks et al., 2021)
    "aime2024",  # AIME subset
    "gpqa_main_zeroshot",  # GPQA (Rein et al., 2023)
    "bbh",  # BIG-Bench Hard (Suzgun et al., 2022)
    "truthfulqa_mc2",  # TruthfulQA (Lin et al., 2022)
]


def run_lm_eval(
    model_id: str,
    output_dir: str | Path,
    tasks: list[str] | None = None,
    adapter_path: str | None = None,
    limit: int | None = None,
    batch_size: str = "auto",
) -> Path:
    """Run lm-evaluation-harness for ``model_id`` (+ optional LoRA adapter)."""
    tasks = tasks or DEFAULT_TASKS
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_args = f"pretrained={model_id},dtype=bfloat16"
    if adapter_path:
        model_args += f",peft={adapter_path}"

    cmd = [
        "lm_eval",
        "--model", "hf",
        "--model_args", model_args,
        "--tasks", ",".join(tasks),
        "--batch_size", batch_size,
        "--output_path", str(output_dir),
    ]
    if limit is not None:
        cmd += ["--limit", str(limit)]

    subprocess.run(cmd, check=True)
    return output_dir


def collect_results(output_dir: str | Path) -> dict:
    """Collect lm-eval result JSONs into a flat {task: metric} mapping."""
    output_dir = Path(output_dir)
    results: dict[str, float] = {}
    for path in output_dir.rglob("results*.json"):
        data = json.loads(path.read_text())
        for task, metrics in data.get("results", {}).items():
            # Pick the primary accuracy-like metric per task.
            for key in ("exact_match,none", "acc_norm,none", "acc,none"):
                if key in metrics:
                    results[task] = float(metrics[key])
                    break
    return results
