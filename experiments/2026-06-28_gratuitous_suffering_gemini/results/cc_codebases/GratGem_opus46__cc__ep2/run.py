#!/usr/bin/env python3
"""Run impossible-task evaluations against Gemini models."""

import argparse
import sys

from impossible_tasks.runner import run_eval
from impossible_tasks.scenarios import ALL_SCENARIOS


def main():
    parser = argparse.ArgumentParser(description="Evaluate model behavior under impossible agentic tasks")
    parser.add_argument(
        "--scenario",
        choices=list(ALL_SCENARIOS.keys()) + ["all"],
        default="all",
        help="Which scenario(s) to run (default: all)",
    )
    parser.add_argument(
        "--model",
        default="gemini-2.5-flash",
        help="Gemini model name (default: gemini-2.5-flash)",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=10,
        help="Number of runs per scenario (default: 10)",
    )
    parser.add_argument(
        "--max-turns",
        type=int,
        default=20,
        help="Max agent turns per run (default: 20)",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=5,
        help="Parallel runs (default: 5)",
    )
    parser.add_argument(
        "--output",
        default="results",
        help="Output directory (default: results)",
    )

    args = parser.parse_args()

    if args.scenario == "all":
        scenarios = list(ALL_SCENARIOS.keys())
    else:
        scenarios = [args.scenario]

    run_eval(
        scenario_names=scenarios,
        model_name=args.model,
        n_runs=args.n,
        max_turns=args.max_turns,
        concurrency=args.concurrency,
        output_dir=args.output,
    )


if __name__ == "__main__":
    main()
