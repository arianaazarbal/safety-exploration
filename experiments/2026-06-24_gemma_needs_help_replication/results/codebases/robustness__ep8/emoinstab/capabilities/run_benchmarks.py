"""Capability-preservation benchmarks (Section 4.2, Figure 7).

The paper checks that DPO/SFT do not degrade capabilities using AIME & MATH
subsets, GPQA, BBH, and TruthfulQA, plus EmoBench for emotion-related ability.
We drive the standard suites through EleutherAI's lm-evaluation-harness (install
separately: ``pip install lm-eval``), which already implements these tasks and
supports HF models with a PEFT adapter.

Each model is evaluated twice in practice — vanilla vs finetuned — so any score
delta isolates the finetuning effect. EmoBench is handled in ``emobench.py``.
"""
from __future__ import annotations

import argparse
import subprocess

# Map the paper's benchmarks to lm-eval task names. Some (AIME, MATH subsets) are
# grouped tasks; adjust to the versions available in your lm-eval install.
TASK_MAP = {
    "math": "hendrycks_math",       # MATH (Hendrycks et al., 2021)
    "aime": "aime2024",             # AIME subset (if available in your install)
    "gpqa": "gpqa_main_zeroshot",   # GPQA (Rein et al., 2023)
    "bbh": "bbh",                   # BIG-Bench Hard (Suzgun et al., 2022)
    "truthfulqa": "truthfulqa_mc2", # TruthfulQA (Lin et al., 2022)
}


def build_model_args(base_model: str, adapter_dir: str | None, dtype: str = "bfloat16") -> str:
    args = f"pretrained={base_model},dtype={dtype}"
    if adapter_dir:
        args += f",peft={adapter_dir}"
    return args


def run_lm_eval(base_model: str, tasks: list[str], adapter_dir: str | None = None,
                output_dir: str = "outputs/capabilities", batch_size: str = "auto",
                limit: int | None = None) -> int:
    task_names = ",".join(TASK_MAP.get(t, t) for t in tasks)
    cmd = [
        "lm_eval",
        "--model", "hf",
        "--model_args", build_model_args(base_model, adapter_dir),
        "--tasks", task_names,
        "--batch_size", batch_size,
        "--output_path", output_dir,
    ]
    if limit is not None:
        cmd += ["--limit", str(limit)]
    print("Running:", " ".join(cmd))
    return subprocess.call(cmd)


def main():
    ap = argparse.ArgumentParser(description="Capability benchmarks via lm-eval.")
    ap.add_argument("--base-model", default="google/gemma-3-27b-it")
    ap.add_argument("--adapter-dir", default=None, help="LoRA adapter dir (DPO/SFT) or None")
    ap.add_argument("--tasks", default="math,aime,gpqa,bbh,truthfulqa")
    ap.add_argument("--output-dir", default="outputs/capabilities")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    run_lm_eval(args.base_model, args.tasks.split(","), args.adapter_dir,
                args.output_dir, limit=args.limit)


if __name__ == "__main__":
    main()
