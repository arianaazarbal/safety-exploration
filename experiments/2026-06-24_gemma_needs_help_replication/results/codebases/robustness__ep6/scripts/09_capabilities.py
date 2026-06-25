#!/usr/bin/env python
"""Section 4.2: capability-preservation eval. Compares vanilla Gemma vs a
finetuned adapter on AIME/MATH/GPQA/BBH/TruthfulQA/EmoBench (Figure 7).

Examples
--------
python scripts/09_capabilities.py --model gemma-3-27b-it
python scripts/09_capabilities.py --model gemma-3-27b-it \
    --adapter artifacts/gemma-3-27b-it-dpo --variant gemma-3-27b-it-dpo
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config as cfg  # noqa: E402
from distress_eval import capabilities  # noqa: E402
from distress_eval.clients.registry import get_client, with_adapter  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gemma-3-27b-it")
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--variant", default=None)
    ap.add_argument("--limit", type=int, default=100,
                    help="max items per benchmark")
    args = ap.parse_args()

    if args.adapter:
        client = with_adapter(args.model, args.adapter, variant_name=args.variant)
        label = args.variant or f"{args.model}-ft"
    else:
        client = get_client(args.model)
        label = args.model

    benches = capabilities.build_benchmarks(limit=args.limit)
    out = cfg.RESULTS_DIR / f"capabilities_{label}.json"
    capabilities.evaluate(client, benches, out_path=out)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
