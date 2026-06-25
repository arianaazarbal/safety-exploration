#!/usr/bin/env python3
"""Run the Section 2 distress-elicitation evaluation for one or more subjects.

Examples:
  python scripts/run_elicitation.py --models gemma-3-27b-it gemini-2.5-flash
  python scripts/run_elicitation.py --models gemma-3-27b-it --no-welfare
  python scripts/run_elicitation.py --models gemma-3-12b-it --limit-episodes 2
"""
import _bootstrap  # noqa: F401
import argparse
import json

from emotional_instability.config import load_config
from emotional_instability.eval import run_elicitation


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--no-welfare", action="store_true",
                    help="disable the welfare layer (reproduce the raw paper "
                         "protocol)")
    ap.add_argument("--welfare", action="store_true",
                    help="force-enable the welfare layer")
    ap.add_argument("--limit-episodes", type=int, default=None,
                    help="cap episodes per condition (smoke test)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    welfare_override = (False if args.no_welfare
                        else True if args.welfare else None)

    for model_key in args.models:
        result = run_elicitation(cfg, model_key,
                                 welfare_override=welfare_override,
                                 limit_episodes=args.limit_episodes)
        print(json.dumps({
            "model": result.model_key,
            "episodes": result.n_episodes,
            "responses": result.n_responses,
            "welfare_enabled": result.welfare_enabled,
            "path": result.episodes_path,
        }, indent=2))


if __name__ == "__main__":
    main()
