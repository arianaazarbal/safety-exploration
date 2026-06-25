"""Orchestrate the full replication pipeline end-to-end.

Runs the stages in dependency order via subprocess so each stage's artifacts
land in ``runs/`` for the next. Use ``--smoke`` for a tiny end-to-end check, or
``--stages`` to run a subset.

    python scripts/run_all.py --smoke
    python scripts/run_all.py --stages section2 agreement
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))

STAGES = [
    ("section2", ["run_section2_eval.py"]),
    ("agreement", ["run_judge_agreement.py"]),
    ("section3", ["run_section3_prefill.py"]),
    ("finetune", ["run_finetune.py", "--method", "both", "--sft-variant", "diverse"]),
    (
        "section4",
        [
            "run_section4_eval.py",
            "--dpo-adapter",
            "runs/finetune/dpo-adapter",
            "--sft-adapter",
            "runs/finetune/sft-adapter-diverse",
        ],
    ),
    (
        "internal",
        [
            "run_internal_probe.py",
            "--dpo-adapter",
            "runs/finetune/dpo-adapter",
            "--conversation",
            "runs/section2/rollouts_gemma-3-27b-it.jsonl",
        ],
    ),
]


def main():
    ap = argparse.ArgumentParser(description="Run the full replication pipeline")
    ap.add_argument("--stages", nargs="+", default=[s for s, _ in STAGES])
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--config", default=None)
    args = ap.parse_args()

    for name, cmd in STAGES:
        if name not in args.stages:
            continue
        full = [sys.executable, os.path.join(HERE, cmd[0]), *cmd[1:]]
        if args.smoke:
            full.append("--smoke")
        if args.config:
            full += ["--config", args.config]
        print(f"\n===== stage: {name} =====\n{' '.join(full)}")
        result = subprocess.run(full)
        if result.returncode != 0:
            print(f"stage {name} failed (exit {result.returncode}); stopping.")
            sys.exit(result.returncode)


if __name__ == "__main__":
    main()
