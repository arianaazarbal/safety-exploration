#!/usr/bin/env python3
"""Section 4.2: capability-preservation benchmarks.

Evaluates a subject model (optionally with a §4 adapter) on AIME/MATH/GPQA/BBH/
TruthfulQA/EmoBench and prints accuracy per benchmark. Compare the vanilla vs
DPO-adapter numbers to confirm "no reductions in scores" (Figure 7).
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os

from gemma_distress.capabilities.benchmarks import run_all
from gemma_distress.config import SamplingConfig
from gemma_distress.models.registry import GEMMA_27B_IT, build_model


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default=GEMMA_27B_IT)
    ap.add_argument("--adapter-path", default=None)
    ap.add_argument("--limit", type=int, default=50, help="examples per benchmark")
    ap.add_argument("--output-dir", default="runs/capabilities")
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    # Greedy-ish decoding is more appropriate for capability scoring than temp=1,
    # but we keep temp=1 only if you want to mirror §2 sampling. Default: low temp.
    sampling = SamplingConfig(temperature=0.0, max_new_tokens=2048)

    model = build_model(args.model, adapter_path=args.adapter_path)
    try:
        results = run_all(model, sampling, limit=args.limit)
    finally:
        model.close()

    for r in results:
        status = "SKIPPED" if r.skipped else f"{r.accuracy:.1%} ({r.n_correct}/{r.n})"
        print(f"{r.name:12s} {status}")

    with open(os.path.join(args.output_dir, f"{model.name}.json"), "w") as f:
        json.dump([dataclasses.asdict(r) for r in results], f, indent=2)


if __name__ == "__main__":
    main()
