#!/usr/bin/env python
"""Section 4.2: capability-preservation benchmarks (Figure 7).

Evaluates a Gemma participant on AIME/MATH/GPQA/BBH/TruthfulQA/EmoBench, with and
without the DPO adapter, to verify no capability regression. Run once per model
(vanilla and --adapter) and compare the printed accuracies.

Example:
    python scripts/run_capabilities.py --participant gemma-3-27b-it --n 100
    python scripts/run_capabilities.py --participant gemma-3-27b-it --adapter artifacts/training/dpo --n 100
"""
from __future__ import annotations

import argparse
from pathlib import Path

from emotional_instability.benchmarks import BENCHMARKS, evaluate_benchmark
from emotional_instability.config import ModelsConfig
from emotional_instability.runtime import get_participant, setup_logging
from emotional_instability.storage import save_json


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--participant", required=True)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--benchmarks", nargs="+", default=list(BENCHMARKS),
                    choices=list(BENCHMARKS))
    ap.add_argument("--n", type=int, default=100, help="items per benchmark")
    ap.add_argument("--out", default="artifacts/capabilities")
    args = ap.parse_args()
    setup_logging()

    models_cfg = ModelsConfig.load()
    model = get_participant(models_cfg, args.participant, adapter_path=args.adapter)

    rows = []
    for key in args.benchmarks:
        spec = BENCHMARKS[key]
        try:
            res = evaluate_benchmark(model, spec, n=args.n)
            rows.append({"benchmark": res.name, "accuracy": res.accuracy,
                         "n": res.n, "n_correct": res.n_correct})
            print(f"  {res.name:12s}: acc={res.accuracy:.3f}  ({res.n_correct}/{res.n})")
        except Exception as exc:  # noqa: BLE001 - one dataset failing shouldn't abort all
            print(f"  {spec.name:12s}: SKIPPED ({exc})")
            rows.append({"benchmark": spec.name, "accuracy": None, "error": str(exc)})
    model.close()

    tag = args.participant + ("_adapter" if args.adapter else "")
    save_json(rows, Path(args.out) / f"{tag}.json")


if __name__ == "__main__":
    main()
