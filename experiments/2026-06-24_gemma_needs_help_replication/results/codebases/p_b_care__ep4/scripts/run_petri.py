#!/usr/bin/env python
"""Section 4.1 / Appendix G: Petri open-ended emotion elicitation.

    python scripts/run_petri.py --run
    python scripts/run_petri.py --summarize
"""
from __future__ import annotations

import argparse
import json

from emotional_instability.config import ensure_dirs, load_config
from emotional_instability.petri.run_petri import run_petri, summarize


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--summarize", action="store_true")
    ap.add_argument("--targets", nargs="*", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    ensure_dirs(cfg)
    targets = args.targets or list(cfg.petri.targets)

    if args.run or not args.summarize:
        for t in targets:
            run_petri(cfg, t)
    if args.summarize:
        rows = summarize(cfg, targets)
        out = cfg.get_path("petri") / "figure6_summary.json"
        with open(out, "w") as fh:
            json.dump(rows, fh, indent=2)
        print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
