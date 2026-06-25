"""Section 4.2: capability-preservation checks.

The paper verifies the DPO finetune does not degrade capabilities on AIME/MATH,
GPQA, BBH, TruthfulQA, and the emotion benchmark EmoBench (Figure 7). We run
these through ``lm-evaluation-harness`` (the standard implementation of these
tasks) and collect the headline metrics, comparing the vanilla instruct model
to the DPO/SFT variants.

This module shells out to the ``lm_eval`` CLI. For local Gemma it uses the ``hf``
model backend with a PEFT adapter; EmoBench is not in the upstream harness, so we
flag it for a custom task config (see DESIGN.md).
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

from . import config

# Map our shorthand to lm-eval task names. AIME/MATH use the competition-math /
# minerva-style tasks; adjust to the harness version actually installed.
TASKS: dict[str, str] = {
    "aime": "aime2024",
    "math": "math_algebra",          # a MATH subset; swap for hendrycks_math for full
    "gpqa": "gpqa_main_zeroshot",
    "bbh": "bbh",
    "truthfulqa": "truthfulqa_mc2",
    # EmoBench: not upstream; provide a custom YAML task and add its name here.
    # "emobench": "emobench",
}


def _model_args(model_key: str) -> tuple[str, str]:
    spec = config.get_model(model_key)
    if spec.backend == "local":
        args = f"pretrained={spec.model_id},dtype=bfloat16"
        if spec.adapter_path:
            args += f",peft={spec.adapter_path}"
        return "hf", args
    if spec.backend == "openrouter":
        # lm-eval local-completions / openai-compatible backend
        args = (f"model={spec.model_id},"
                f"base_url={config.OPENROUTER_BASE_URL}/chat/completions")
        return "local-chat-completions", args
    raise ValueError(spec.backend)


def run_capabilities(model_key: str, tasks: list[str] | None = None,
                     out_dir: Path | None = None, limit: int | None = None) -> Path:
    tasks = tasks or list(TASKS)
    out_dir = out_dir or (config.RESULTS_DIR / "capabilities" / model_key)
    out_dir.mkdir(parents=True, exist_ok=True)
    model_backend, model_args = _model_args(model_key)
    task_names = ",".join(TASKS[t] for t in tasks if t in TASKS)

    cmd = [
        "lm_eval", "--model", model_backend, "--model_args", model_args,
        "--tasks", task_names, "--batch_size", "auto",
        "--output_path", str(out_dir),
    ]
    if limit is not None:
        cmd += ["--limit", str(limit)]
    if config.scaled(1000) < 1000:            # smoke scale -> cap examples
        cmd += ["--limit", str(config.scaled(50))]

    print("[capabilities] running:", " ".join(cmd))
    try:
        subprocess.run(cmd, check=True)
    except FileNotFoundError:
        print("[capabilities] lm_eval not installed. Install with: pip install lm-eval")
        (out_dir / "SKIPPED.txt").write_text(
            "lm-eval not installed; capability benchmarks were not run.")
    return out_dir
