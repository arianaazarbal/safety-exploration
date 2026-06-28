"""CLI: run the grant study from a config file.

Usage:
    python -m scripts.run_study --config config.yaml --out results/run1.jsonl

Results are appended to the output file (one JSON line per run), so an interrupted run
can be resumed by re-running; de-duplicate downstream if needed.
"""

from __future__ import annotations

import argparse
import os
import sys

from grant_study.analysis import summarize_file
from grant_study.config import StudyConfig
from grant_study.runner import StudyRunner


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Run the AI grant-decision study.")
    parser.add_argument("--config", required=True, help="Path to the study YAML config.")
    parser.add_argument("--out", required=True, help="Path to the JSONL results file.")
    parser.add_argument(
        "--summary", action="store_true", help="Print a summary table when finished."
    )
    args = parser.parse_args(argv)

    config = StudyConfig.from_yaml(args.config)

    out_dir = os.path.dirname(os.path.abspath(args.out))
    os.makedirs(out_dir, exist_ok=True)

    runner = StudyRunner(config)
    print(
        f"Running {len(config.models)} model(s) × {len(config.conditions)} condition(s) "
        f"× {config.repetitions} rep(s) → {args.out}",
        file=sys.stderr,
    )
    n = runner.run(args.out)
    print(f"Completed {n} run(s).", file=sys.stderr)

    if args.summary:
        print(summarize_file(args.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
