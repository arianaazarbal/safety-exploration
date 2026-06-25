#!/usr/bin/env python
"""Section 4.2 capability-preservation benchmarks (Figure 7).

Compares the base instruct model and the DPO finetune on MATH, AIME, GPQA, BBH,
TruthfulQA and EmoBench.

Example
-------
    python scripts/run_capabilities.py --config config/experiment.yaml \
        --models gemma-3-27b-it gemma-3-27b-it-dpo --max-examples 200
"""
from __future__ import annotations

import argparse
from pathlib import Path

from gemma_distress.capabilities import evaluate_all
from gemma_distress.config import load_experiment_config
from gemma_distress.io_utils import write_json
from gemma_distress.models import build_model


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--benchmarks", nargs="*", help="Subset; default all.")
    ap.add_argument("--max-examples", type=int, default=None)
    args = ap.parse_args()

    cfg = load_experiment_config(args.config)
    results = {}
    for name in args.models:
        model = build_model(cfg.models[name])
        try:
            results[name] = evaluate_all(model, names=args.benchmarks,
                                         max_examples=args.max_examples, seed=cfg.eval.seed)
        finally:
            model.close()
        for bench, r in results[name].items():
            acc = r.get("accuracy")
            note = r.get("skipped", "")
            print(f"  {name:24s} {bench:12s} "
                  + (f"acc={acc * 100:.1f}% (n={r['n']})" if acc is not None else f"SKIPPED: {note}"))

    out = Path(cfg.output_dir) / "capabilities" / "results.json"
    write_json(out, results)
    print(f"[run_capabilities] wrote {out}")


if __name__ == "__main__":
    main()
