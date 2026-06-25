"""Capability-preservation evaluation (Section 4.2, Figure 7).

The paper checks that DPO/SFT do not degrade capabilities on AIME, MATH, GPQA,
BBH, TruthfulQA and emotion capabilities on EmoBench. The standard, least-error-
prone way to run these is the EleutherAI lm-evaluation-harness, which already
implements all but EmoBench. We shell out to it (with the LoRA adapter merged or
loaded via peft) and collect the headline metric per task.

This is intentionally a thin wrapper: re-implementing six benchmarks by hand
would be a large surface for subtle scoring bugs. EmoBench is provided as a
custom path since it is not in the harness by default (see DESIGN.md).
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

from ..config import (CAPABILITY_BENCHMARKS, FINETUNE_BASE_MODEL, MODELS,
                      RESULTS_DIR)


def build_harness_command(model_id: str, tasks: list[str], *,
                          adapter_path: str | None = None,
                          limit: int | None = None,
                          batch_size: str = "auto",
                          output_path: Path | None = None) -> list[str]:
    model_args = f"pretrained={model_id},dtype=bfloat16"
    if adapter_path:
        model_args += f",peft={adapter_path}"
    cmd = [
        "lm_eval",
        "--model", "hf",
        "--model_args", model_args,
        "--tasks", ",".join(tasks),
        "--batch_size", batch_size,
    ]
    if limit:
        cmd += ["--limit", str(limit)]
    if output_path:
        cmd += ["--output_path", str(output_path)]
    return cmd


def run_capabilities(model_key: str = FINETUNE_BASE_MODEL,
                     *, adapter_path: str | None = None,
                     benchmarks: list[str] | None = None,
                     dry_run: bool = False) -> Path:
    """Run the capability suite for one (model, adapter). Returns results path."""
    model_id = MODELS[model_key].model_id
    benchmarks = benchmarks or list(CAPABILITY_BENCHMARKS)

    label = model_key if not adapter_path else f"{model_key}+adapter"
    out_dir = RESULTS_DIR / f"capabilities_{label.replace('/', '_')}"
    out_dir.mkdir(parents=True, exist_ok=True)

    results = {}
    for name in benchmarks:
        if name not in CAPABILITY_BENCHMARKS:
            print(f"[capabilities] unknown benchmark {name}, skipping")
            continue
        task_id, limit = CAPABILITY_BENCHMARKS[name]
        if name == "emobench":
            print("[capabilities] EmoBench is not in lm-eval-harness by default; "
                  "see capability/emobench.py for the custom runner.")
            continue
        cmd = build_harness_command(model_id, [task_id], adapter_path=adapter_path,
                                    limit=limit, output_path=out_dir / name)
        print("[capabilities] " + " ".join(cmd))
        if not dry_run:
            subprocess.run(cmd, check=True)
        results[name] = {"task": task_id, "limit": limit,
                         "output": str(out_dir / name)}

    summary_path = out_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump({"model": label, "benchmarks": results}, f, indent=2)
    print(f"[capabilities] summary -> {summary_path}")
    return summary_path
