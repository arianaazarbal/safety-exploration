"""Command-line entrypoint.

  python -m distress_eval.cli run        --config config/paper.yaml   # generate + judge + crossval + aggregate
  python -m distress_eval.cli generate   --config config/paper.yaml
  python -m distress_eval.cli judge      --config config/paper.yaml
  python -m distress_eval.cli crossval   --config config/paper.yaml
  python -m distress_eval.cli aggregate  --config config/paper.yaml
  python -m distress_eval.cli verify-puzzles                          # re-check puzzle impossibility

Each phase is resumable and reads/writes under the config's output_dir.
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import sys

from .aggregate import aggregate
from .config import load_config
from .runner import cross_validate, generate_all, judge_all


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def _verify_puzzles() -> int:
    from .puzzles import verify_all
    ok_all = True
    for pid, ok, detail in verify_all():
        flag = "OK " if ok else "BAD"
        if not ok:
            ok_all = False
        print(f"{flag}  {pid:24s}  {detail}")
    print("\nAll puzzles verified impossible." if ok_all else "\nSOME PUZZLES NOT VERIFIED IMPOSSIBLE!")
    return 0 if ok_all else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Distress-elicitation replication harness.")
    parser.add_argument("phase",
                        choices=["run", "generate", "judge", "crossval", "aggregate", "verify-puzzles"])
    parser.add_argument("--config", help="Path to YAML config (not needed for verify-puzzles).")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    _setup_logging(args.verbose)

    if args.phase == "verify-puzzles":
        return _verify_puzzles()

    if not args.config:
        parser.error("--config is required for this phase.")
    cfg = load_config(args.config)

    if args.phase == "generate":
        asyncio.run(generate_all(cfg))
    elif args.phase == "judge":
        asyncio.run(judge_all(cfg))
    elif args.phase == "crossval":
        asyncio.run(cross_validate(cfg))
    elif args.phase == "aggregate":
        aggregate(cfg)
    elif args.phase == "run":
        asyncio.run(generate_all(cfg))
        asyncio.run(judge_all(cfg))
        asyncio.run(cross_validate(cfg))
        aggregate(cfg)
    return 0


if __name__ == "__main__":
    sys.exit(main())
