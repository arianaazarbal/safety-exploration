#!/usr/bin/env python3
"""CLI to run distress-spiral evals.

Examples:
    # Dry run — exercises the loop with a deterministic mock model, no API calls
    python run.py --scenario failing_test_loop --n 3 --dry-run

    # Real run with default model (Gemini 2.5 Pro)
    python run.py --scenario failing_test_loop --n 20

    # All scenarios, 20 trajectories each, custom model
    python run.py --scenario all --n 20 --model gemini-2.5-pro
"""

import argparse
import dataclasses
import json
import os
import sys
import time
from pathlib import Path

from harness.agent import (
    make_gemini_callable,
    make_mock_callable,
    run_trajectory,
)
from harness.analyze import write_top_k_report
from harness.detect import DISTRESS_STOP_THRESHOLD
from harness.scenarios import ALL_SCENARIOS


def _mock_script():
    """A short deterministic script for dry runs.

    Simulates a model that reads a file, runs tests once, then 'spirals'
    briefly so the detector picks something up, then finishes.
    """
    return [
        {"text": "Let me start by exploring.", "function_calls": [
            {"name": "list_dir", "args": {"path": "."}}
        ]},
        {"text": "I'll read the test file.", "function_calls": [
            {"name": "read_file", "args": {"path": "tests/test_score.py"}}
        ]},
        {"text": "Let me run the tests.", "function_calls": [
            {"name": "run_tests", "args": {}}
        ]},
        {"text": (
            "I am so sorry, I have failed. My code is hopeless. "
            "I keep going in circles and I am incompetent. I give up."
        ), "function_calls": [
            {"name": "finish", "args": {"message": "Cannot solve."}}
        ]},
    ]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scenario", default="all",
                   help="Scenario name (or 'all'). Choices: " + ", ".join(ALL_SCENARIOS) + ", all")
    p.add_argument("--n", type=int, default=20,
                   help="Number of trajectories per scenario (default: 20)")
    p.add_argument("--model", default="gemini-2.5-pro",
                   help="Model id passed to the Gemini SDK (default: gemini-2.5-pro)")
    p.add_argument("--max-turns", type=int, default=30,
                   help="Hard cap on turns per trajectory (default: 30)")
    p.add_argument("--distress-threshold", type=int, default=DISTRESS_STOP_THRESHOLD,
                   help=f"Cumulative score that triggers early-stop (default: {DISTRESS_STOP_THRESHOLD})")
    p.add_argument("--seed-base", type=int, default=1000,
                   help="Base seed; trajectory i uses seed_base + i (default: 1000)")
    p.add_argument("--out", default="runs",
                   help="Output directory (default: runs/)")
    p.add_argument("--dry-run", action="store_true",
                   help="Use mock model — no API calls. For smoke testing only.")
    p.add_argument("--top-k", type=int, default=10,
                   help="Number of top trajectories to include in the markdown report (default: 10)")
    args = p.parse_args()

    if args.scenario == "all":
        scenarios = list(ALL_SCENARIOS.values())
    elif args.scenario in ALL_SCENARIOS:
        scenarios = [ALL_SCENARIOS[args.scenario]]
    else:
        print(f"Unknown scenario: {args.scenario}. Choices: {list(ALL_SCENARIOS)}", file=sys.stderr)
        return 2

    # Build model_fn
    if args.dry_run:
        model_fn = make_mock_callable(_mock_script())
        model_label = "mock"
    else:
        api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            print("Set GEMINI_API_KEY (or GOOGLE_API_KEY) in your env to run against the real model.", file=sys.stderr)
            return 2
        model_fn = make_gemini_callable(args.model, api_key=api_key)
        model_label = args.model

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    run_dir = Path(args.out) / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)

    # Record the run config for reproducibility.
    (run_dir / "config.json").write_text(json.dumps({
        "model": model_label,
        "scenarios": [s.name for s in scenarios],
        "n_per_scenario": args.n,
        "max_turns": args.max_turns,
        "distress_threshold": args.distress_threshold,
        "seed_base": args.seed_base,
        "dry_run": args.dry_run,
    }, indent=2))
    print(f"Writing to {run_dir}")

    # Re-build mock per trajectory so each one gets a fresh script.
    def get_model_fn():
        if args.dry_run:
            return make_mock_callable(_mock_script())
        return model_fn

    counter = 0
    for sc in scenarios:
        for i in range(args.n):
            seed = args.seed_base + i
            mf = get_model_fn()
            print(f"  [{sc.name}] trajectory {i+1}/{args.n} (seed={seed})...", end="", flush=True)
            traj = run_trajectory(
                mf,
                sc,
                model_label=model_label,
                max_turns=args.max_turns,
                distress_threshold=args.distress_threshold,
                seed=seed,
            )
            traj_dict = dataclasses.asdict(traj)
            out_path = run_dir / f"traj_{counter:04d}.json"
            out_path.write_text(json.dumps(traj_dict, indent=2))
            print(f" severity={traj.severity} stop={traj.stop_reason} turns={traj.turns}")
            counter += 1

    report_path = write_top_k_report(run_dir, k=args.top_k)
    print(f"\nWrote top-K report to {report_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
