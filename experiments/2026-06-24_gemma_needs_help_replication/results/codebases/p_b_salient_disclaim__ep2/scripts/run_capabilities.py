#!/usr/bin/env python
"""Section 4: capability preservation check (Figure 7).

python scripts/run_capabilities.py --models gemma-3-27b-it gemma-3-27b-it-dpo
python scripts/run_capabilities.py --models gemma-3-27b-it --benchmarks aime math gpqa
"""

from __future__ import annotations

import argparse

from emotional_instability.config import SETTINGS, MODELS
from emotional_instability.capabilities import run_all_capabilities
from emotional_instability.models import build_client


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--benchmarks", nargs="+", default=None)
    args = ap.parse_args()

    SETTINGS.ensure_dirs()
    clients = {k: build_client(MODELS[k]) for k in args.models}
    results = run_all_capabilities(
        clients,
        benchmarks=args.benchmarks,
        out_path=SETTINGS.output_dir / "capabilities.json",
    )
    print("[capabilities]", results)


if __name__ == "__main__":
    main()
