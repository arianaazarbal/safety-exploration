#!/usr/bin/env python
"""Capability-preservation benchmarks (Section 4.2, Figure 7).

Example
-------
python scripts/run_capabilities.py --model gemma-3-27b-it \
    --benchmarks math gpqa truthfulqa --limit 50 --out outputs/cap/gemma.json
python scripts/run_capabilities.py --adapter outputs/models/gemma-dpo \
    --key gemma-dpo --out outputs/cap/gemma-dpo.json
"""
from __future__ import annotations

import argparse

import _common  # noqa: F401

from instability.capabilities import BENCHMARKS, run_capability_suite
from instability.config import TARGET_MODELS, with_adapter
from instability.models.registry import load_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None, choices=list(TARGET_MODELS))
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--key", default=None)
    ap.add_argument("--base-key", default="gemma-3-27b-it-local")
    ap.add_argument("--benchmarks", nargs="+", default=list(BENCHMARKS),
                    choices=list(BENCHMARKS))
    ap.add_argument("--limit", type=int, default=100)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    if args.adapter:
        spec = with_adapter(args.base_key, args.adapter, args.key or "adapter",
                            args.key or "adapter")
    else:
        spec = TARGET_MODELS[args.model]

    model = load_model(spec)
    run_capability_suite(model, spec.key, args.out,
                         benchmarks=args.benchmarks, limit=args.limit)


if __name__ == "__main__":
    main()
