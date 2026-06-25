#!/usr/bin/env python3
"""Section 3: base-vs-instruct prefill experiment (Gemma only).

Example
-------
    python scripts/run_prefill.py --models gemma-3-27b-pt gemma-3-27b-it
"""

from __future__ import annotations

import argparse
import json

from emotional_instability.config import load_config
from emotional_instability.prefill.run import run_prefill, run_recovery


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=None)
    ap.add_argument(
        "--models",
        nargs="+",
        default=["gemma-3-27b-pt", "gemma-3-27b-it"],
        help="Local Gemma base/instruct models to compare via prefill",
    )
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument(
        "--recovery",
        action="store_true",
        help="Run the §4.2 recovery experiment (truncate score>=7 responses near "
        "their end) instead of the base-vs-instruct comparison.",
    )
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.recovery:
        summary = run_recovery(cfg, model_names=args.models, batch_size=args.batch_size)
    else:
        summary = run_prefill(cfg, model_names=args.models, batch_size=args.batch_size)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
