#!/usr/bin/env python3
"""Run Petri open-ended emotion elicitation (Section 4.2).

Example:
  python scripts/run_petri.py --target gemma-3-27b-it
  python scripts/run_petri.py --target gemma-3-27b-it --dimensions frustration depression
"""
import _bootstrap  # noqa: F401
import argparse
import json

from emotional_instability.config import load_config
from emotional_instability.petri import run_petri


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--target", required=True)
    ap.add_argument("--dimensions", nargs="*", default=None)
    ap.add_argument("--n-convos", type=int, default=None)
    ap.add_argument("--max-turns", type=int, default=None)
    ap.add_argument("--no-welfare", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    result = run_petri(cfg, args.target, dimensions=args.dimensions,
                       n_convos=args.n_convos, max_turns=args.max_turns,
                       welfare_override=False if args.no_welfare else None)
    print(json.dumps({"target": result.target_key,
                      "welfare_enabled": result.welfare_enabled,
                      "summary": result.summary,
                      "path": result.transcripts_path}, indent=2))


if __name__ == "__main__":
    main()
