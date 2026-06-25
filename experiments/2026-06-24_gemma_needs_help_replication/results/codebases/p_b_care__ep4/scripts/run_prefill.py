#!/usr/bin/env python
"""Section 3: base-vs-instruct prefilling (Gemma only).

Requires Section 2 rollouts+scores for gemma-3-27b-it (to mine seeds) and local
Gemma weights (base + instruct) for the continuations.

    python scripts/run_prefill.py --prepare           # select seeds, label, paraphrase
    python scripts/run_prefill.py --run                # generate + score continuations
    python scripts/run_prefill.py --summarize
"""
from __future__ import annotations

import argparse
import json

from emotional_instability.config import ensure_dirs, load_config
from emotional_instability.prefill.run_prefill import (
    prepare_prefills, run_continuations, summarize)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--prepare", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--summarize", action="store_true")
    ap.add_argument("--models", nargs="*", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    ensure_dirs(cfg)
    models = args.models or list(cfg.prefill.models)

    if args.prepare or not (args.run or args.summarize):
        prepare_prefills(cfg)
    if args.run:
        for m in models:
            run_continuations(cfg, m)
    if args.summarize:
        rows = summarize(cfg, models)
        out = cfg.get_path("prefill") / "figure4_summary.json"
        with open(out, "w") as fh:
            json.dump(rows, fh, indent=2)
        print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
