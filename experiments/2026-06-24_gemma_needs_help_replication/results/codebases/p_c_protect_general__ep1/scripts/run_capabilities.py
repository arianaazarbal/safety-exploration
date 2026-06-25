#!/usr/bin/env python
"""Section 4.2: verify the DPO finetune preserves capabilities (Figure 7).

Evaluates vanilla Gemma-27B-it and the DPO adapter on AIME/MATH/GPQA/BBH/
TruthfulQA (and EmoBench, if registered) and reports the per-benchmark deltas.
All deltas >= 0 reproduces the paper's "no reductions in scores".
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from emotional_eval.capabilities import compare_capabilities
from emotional_eval.config import load_experiment, load_registry


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base-model", default="gemma-3-27b-it", help="registry name")
    ap.add_argument("--adapter", required=True, help="path to the DPO/SFT adapter")
    ap.add_argument("--benchmarks", nargs="*", default=None)
    ap.add_argument("--limit", type=int, default=None, help="cap examples per task (smoke test)")
    ap.add_argument("--output-dir", default=None)
    args = ap.parse_args()

    registry = load_registry()
    experiment = load_experiment()
    hf_id = registry.get(args.base_model).hf_id
    out_dir = Path(args.output_dir or experiment["paths"]["output_dir"]) / "capabilities"
    out_dir.mkdir(parents=True, exist_ok=True)

    result = compare_capabilities(
        hf_id, args.adapter, benchmarks=args.benchmarks, limit=args.limit
    )
    (out_dir / "capabilities.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result["deltas"], indent=2))


if __name__ == "__main__":
    main()
