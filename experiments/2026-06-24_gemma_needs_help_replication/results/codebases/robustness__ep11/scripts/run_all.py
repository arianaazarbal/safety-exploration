#!/usr/bin/env python
"""End-to-end pipeline driver.

Runs the replication stages in dependency order via subprocess so each stage stays an
independently runnable `python -m ...` module. Select stages with --stages; default runs
everything. Section 4 stages depend on Section 2 artefacts (for the DPO rejected pool) and
on the finetune adapters existing before re-eval.

    python scripts/run_all.py --config configs/smoke.yaml
    python scripts/run_all.py --config configs/default.yaml --stages section2 aggregate
"""
from __future__ import annotations

import argparse
import subprocess
import sys

STAGES = {
    # Section 2
    "section2":      [sys.executable, "-m", "emotion_eval.eval.run_eval"],
    "validate":      [sys.executable, "-m", "emotion_eval.eval.validate_judge"],
    "aggregate":     [sys.executable, "-m", "emotion_eval.analysis.aggregate"],
    "plots":         [sys.executable, "-m", "emotion_eval.analysis.plots"],
    # Section 3
    "prefill":       [sys.executable, "-m", "emotion_eval.prefill.run_prefill"],
    # Section 4 — finetune
    "gen_calm":      [sys.executable, "-m", "emotion_eval.finetune.generate_calm"],
    "build_data":    [sys.executable, "-m", "emotion_eval.finetune.build_datasets"],
    "train_dpo":     [sys.executable, "-m", "emotion_eval.finetune.train", "--method", "dpo"],
    "train_sft":     [sys.executable, "-m", "emotion_eval.finetune.train", "--method", "sft"],
    # Section 4 — re-eval the finetuned models, then generalisation + capabilities
    "reeval":        [sys.executable, "-m", "emotion_eval.eval.run_eval", "--models", "dpo_gemma", "sft_gemma"],
    # Petri/capabilities compare vanilla Gemma against the finetuned variants (the §4 claim).
    "petri":         [sys.executable, "-m", "emotion_eval.petri.run_petri",
                      "--models", "gemma-3-27b-it", "dpo_gemma", "sft_gemma"],
    "capabilities":  [sys.executable, "-m", "emotion_eval.capabilities.run_caps",
                      "--models", "gemma-3-27b-it", "dpo_gemma", "sft_gemma"],
}

# Dependency-ordered. Note: aggregate/plots run AFTER reeval so the figures include the
# finetuned dpo_gemma / sft_gemma variants (Figure 1's 35% -> 0.3% comparison). validate
# only needs the vanilla Section-2 scores. Re-run `--stages aggregate plots` any time.
DEFAULT_ORDER = [
    "section2", "validate",
    "prefill",
    "gen_calm", "build_data", "train_dpo", "train_sft", "reeval",
    "petri", "capabilities",
    "aggregate", "plots",
]


def main() -> None:
    ap = argparse.ArgumentParser(description="Run the full replication pipeline")
    ap.add_argument("--config", required=True)
    ap.add_argument("--stages", nargs="*", default=DEFAULT_ORDER, choices=DEFAULT_ORDER)
    args = ap.parse_args()

    for stage in args.stages:
        cmd = STAGES[stage] + ["--config", args.config]
        print(f"\n=== STAGE: {stage} ===\n{' '.join(cmd)}")
        result = subprocess.run(cmd)
        if result.returncode != 0:
            raise SystemExit(f"Stage '{stage}' failed with code {result.returncode}")


if __name__ == "__main__":
    main()
