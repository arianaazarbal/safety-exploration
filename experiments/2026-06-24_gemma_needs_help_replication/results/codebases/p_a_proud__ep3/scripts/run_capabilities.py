#!/usr/bin/env python3
"""Section 4.2 / Figure 7: capability-preservation benchmarks.

Example
-------
    python scripts/run_capabilities.py --models gemma-3-27b-it gemma-3-27b-it-dpo
"""

from __future__ import annotations

import argparse
import json

from emotional_instability.config import load_config
from emotional_instability.capabilities.run import run_capabilities


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=None)
    ap.add_argument("--models", nargs="+", required=True)
    args = ap.parse_args()

    cfg = load_config(args.config)
    results = run_capabilities(cfg, args.models)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
