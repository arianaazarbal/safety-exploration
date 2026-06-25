#!/usr/bin/env python
"""Section 4: capability-preservation benchmarks (Figure 7).

Compares vanilla Gemma-27B-it vs the DPO finetune on MATH/AIME/GPQA/BBH/
TruthfulQA/EmoBench. Confirms "no reduction in scores".
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from emotional_instability.config import ARTIFACTS_DIR, register_adapter_model  # noqa: E402
from emotional_instability.capabilities import run_capabilities, BENCHMARKS  # noqa: E402
from emotional_instability.analysis import load_jsonl  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+",
                    default=["gemma-3-27b-it", "gemma-3-27b-it-dpo"])
    ap.add_argument("--benchmarks", nargs="+", default=list(BENCHMARKS))
    ap.add_argument("--n", type=int, default=100)
    args = ap.parse_args()

    if "gemma-3-27b-it-dpo" in args.models:
        register_adapter_model("gemma-3-27b-it-dpo", "gemma-3-27b-it",
                               str(ARTIFACTS_DIR / "adapters" / "dpo"))

    path = run_capabilities(args.models, n_per_bench=args.n,
                            benchmarks=args.benchmarks)
    print(f"wrote {path}")
    print(load_jsonl(path).to_string(index=False))


if __name__ == "__main__":
    main()
