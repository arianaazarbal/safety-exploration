#!/usr/bin/env python
"""Capability-preservation benchmarks (Section 4.2, Figure 7).

Evaluates a model (optionally with a LoRA adapter) on AIME, MATH, GPQA, BBH,
TruthfulQA and EmoBench, to confirm DPO does not degrade capabilities. Compare the
vanilla and DPO models by running twice and diffing the JSON.

python scripts/run_benchmarks.py --model gemma-3-27b-it --out-dir results/bench
python scripts/run_benchmarks.py --model gemma-3-27b-it --adapter checkpoints/dpo \
    --out-dir results/bench
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from emotional_instability.benchmarks import BENCHMARKS, evaluate_benchmark  # noqa: E402
from emotional_instability.models.registry import load_model  # noqa: E402
from emotional_instability.utils.io import load_config, write_jsonl  # noqa: E402
from emotional_instability.utils.seeding import seed_everything  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--benchmarks", nargs="+", default=None,
                    help=f"Subset of {list(BENCHMARKS)} (default: config)")
    ap.add_argument("--max-examples", type=int, default=None)
    ap.add_argument("--out-dir", default="results/bench")
    args = ap.parse_args()

    seed_everything(0)
    names = args.benchmarks or load_config("training")["capabilities"]["benchmarks"]
    model = load_model(args.model, adapter_path=args.adapter)

    results = []
    for name in names:
        res = evaluate_benchmark(model, name, max_examples=args.max_examples)
        print(f"{name:12s} acc={res['accuracy']:.3f} (n={res['n']})")
        results.append({k: v for k, v in res.items() if k != "records"})

    out_dir = Path(args.out_dir)
    tag = model.name.replace("/", "_")
    write_jsonl(out_dir / f"bench_{tag}.jsonl", results)


if __name__ == "__main__":
    main()
