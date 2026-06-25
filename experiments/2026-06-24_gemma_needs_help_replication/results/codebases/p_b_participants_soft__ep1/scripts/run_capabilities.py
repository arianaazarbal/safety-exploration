#!/usr/bin/env python3
"""Capability-preservation benchmarks (Section 4.2 / Figure 7): compare a base
participant against its DPO finetune on AIME/MATH/GPQA/BBH/TruthfulQA/EmoBench.

Example
-------
    python scripts/run_capabilities.py --models gemma-3-27b-it dpo-gemma \
        --adapter checkpoints/dpo --benchmarks math gpqa emobench --n 100
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from emotional_instability import config  # noqa: E402
from emotional_instability.capabilities import run_benchmarks  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", default=["gemma-3-27b-it"])
    parser.add_argument("--adapter", default=None,
                        help="If a model name is unregistered, treat it as a finetune of "
                             "--base using this adapter path.")
    parser.add_argument("--base", default=config.SOURCE_MODEL)
    parser.add_argument("--benchmarks", nargs="+", default=None)
    parser.add_argument("--n", type=int, default=100)
    args = parser.parse_args()

    config.ensure_dirs()
    out_root = config.RESULTS_DIR / "capabilities"
    out_root.mkdir(parents=True, exist_ok=True)

    for model_name in args.models:
        if model_name not in config.PARTICIPANTS and args.adapter:
            config.register_finetune(model_name, args.adapter, base=args.base)
        print(f"== Capabilities: {model_name} ==", flush=True)
        results = run_benchmarks.evaluate_model(
            model_name, benchmarks=args.benchmarks, n=args.n
        )
        (out_root / f"{model_name}.json").write_text(json.dumps(results, indent=2))
        print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
