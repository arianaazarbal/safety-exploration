#!/usr/bin/env python3
"""Section 4.2: capability-preservation check (Figure 7).

Compares vanilla vs DPO Gemma on math/reasoning/truthfulness/emotion benchmarks
to confirm the intervention does not degrade capabilities.
"""

from __future__ import annotations

import argparse

import pandas as pd

from emotional_instability.capability import BENCHMARKS, compare
from emotional_instability.config import load_config


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=None)
    ap.add_argument("--models", nargs="+", default=["gemma-3-27b-it", "gemma-dpo"])
    ap.add_argument("--benchmarks", nargs="*", default=list(BENCHMARKS.keys()))
    args = ap.parse_args()

    cfg = load_config(args.config)
    rows = compare(cfg, args.models, args.benchmarks)
    df = pd.DataFrame(rows)
    print(df.pivot(index="benchmark", columns="model", values="accuracy").to_string())


if __name__ == "__main__":
    main()
