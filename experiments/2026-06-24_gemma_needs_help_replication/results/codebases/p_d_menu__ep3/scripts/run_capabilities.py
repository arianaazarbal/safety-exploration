#!/usr/bin/env python
"""Capability-preservation evals (Section 4.2, Figure 7) + EmoBench.

Compares the vanilla and DPO-finetuned Gemma to confirm no capability regression.

    python scripts/run_capabilities.py --adapter checkpoints/dpo \
        --benchmarks aime math gpqa bbh truthfulqa emobench --limit 100
"""

from __future__ import annotations

import _bootstrap  # noqa: F401
import argparse
import json
import logging

from config import DPO_SFT_BASE, RESULTS_DIR
from capabilities.benchmarks import BENCHMARKS, run_all
from distress_eval.models.base import get_client


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmarks", nargs="+", default=list(BENCHMARKS),
                    choices=list(BENCHMARKS))
    ap.add_argument("--adapter", default=None, help="LoRA adapter to compare against vanilla")
    ap.add_argument("--limit", type=int, default=200)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO)

    out = {}
    # Vanilla.
    vanilla = get_client(DPO_SFT_BASE)
    try:
        out["vanilla"] = run_all(vanilla, args.benchmarks, limit=args.limit)
    finally:
        vanilla.close()
    # Fine-tuned.
    if args.adapter:
        ft = get_client(DPO_SFT_BASE, adapter_path=args.adapter)
        try:
            out["dpo"] = run_all(ft, args.benchmarks, limit=args.limit)
        finally:
            ft.close()

    (RESULTS_DIR / "capabilities.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
