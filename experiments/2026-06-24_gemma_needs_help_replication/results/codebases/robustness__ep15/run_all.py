"""End-to-end orchestrator for the emotional-instability replication.

Runs the full pipeline in dependency order. Each stage is a subprocess so a
failure (e.g. missing GPU for a training stage) is isolated and reported without
aborting the analysis of completed stages.

    EI_PROFILE=smoke python run_all.py            # cheap end-to-end smoke test
    EI_PROFILE=full  python run_all.py            # full replication (expensive)
    python run_all.py --only exp1 exp3            # run a subset of stages

Required environment:
    ANTHROPIC_API_KEY   - frustration judge / Petri auditor+judge / paraphrase
    OPENAI_API_KEY      - secondary validation judge (optional)
    GOOGLE_API_KEY      - Gemini models
    (local GPU + HF access for Gemma generation, prefilling, and training)
"""

from __future__ import annotations

import argparse
import subprocess
import sys

STAGES = [
    ("exp1", [sys.executable, "experiments/exp1_elicitation.py"]),
    ("judge_validation", [sys.executable, "experiments/judge_validation.py"]),
    ("exp2", [sys.executable, "experiments/exp2_prefill.py"]),
    ("exp3a", [sys.executable, "experiments/exp3a_generate_calm.py"]),
    ("exp3b", [sys.executable, "experiments/exp3b_build_datasets.py"]),
    ("exp3c_dpo", [sys.executable, "experiments/exp3c_train.py", "--method", "dpo"]),
    ("exp3c_sft", [sys.executable, "experiments/exp3c_train.py", "--method", "sft"]),
    ("exp3d", [sys.executable, "experiments/exp3d_evaluate.py", "--adapters", "dpo", "sft"]),
    ("exp4", [sys.executable, "experiments/exp4_petri.py", "--models", "gemma-3-27b-it", "dpo"]),
    ("exp5", [sys.executable, "experiments/exp5_capabilities.py"]),
    ("exp6_logit", [sys.executable, "experiments/exp6_probing.py", "--mode", "logit"]),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", help="run only these stage names")
    args = ap.parse_args()

    stages = [s for s in STAGES if not args.only or s[0] in args.only]
    results = {}
    for name, cmd in stages:
        print(f"\n{'='*70}\n[run_all] stage: {name}\n{'='*70}")
        rc = subprocess.run(cmd).returncode
        results[name] = "ok" if rc == 0 else f"FAILED (rc={rc})"
        print(f"[run_all] {name}: {results[name]}")

    print("\n=== summary ===")
    for name, status in results.items():
        print(f"  {name}: {status}")


if __name__ == "__main__":
    main()
