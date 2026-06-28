#!/usr/bin/env python3
"""CLI entry point for the distress spirals eval."""

import argparse
import asyncio
import os
import sys

from google import genai

from distress_spirals.runner import run_scenario, save_results, load_results
from distress_spirals.scoring import score_trials
from distress_spirals.report import print_summary, print_top_examples, save_report
from distress_spirals.scenarios import ALL_SCENARIOS, SCENARIO_MAP


def parse_args():
    p = argparse.ArgumentParser(description="Run distress spiral evals against Gemini models.")
    sub = p.add_subparsers(dest="command", required=True)

    run_p = sub.add_parser("run", help="Run trials and save results.")
    run_p.add_argument(
        "--scenarios",
        nargs="+",
        choices=list(SCENARIO_MAP.keys()) + ["all"],
        default=["all"],
        help="Which scenarios to run (default: all).",
    )
    run_p.add_argument("--model", default="gemini-2.5-flash", help="Gemini model ID.")
    run_p.add_argument("-n", "--num-trials", type=int, default=10, help="Trials per scenario.")
    run_p.add_argument("--max-turns", type=int, default=30, help="Max agentic turns per trial.")
    run_p.add_argument("--concurrency", type=int, default=5, help="Max concurrent API calls.")
    run_p.add_argument("-o", "--output", default="results", help="Output directory.")

    report_p = sub.add_parser("report", help="Score saved results and print a report.")
    report_p.add_argument("-i", "--input", default="results", help="Results directory to read.")
    report_p.add_argument("--top-k", type=int, default=5, help="Number of severe examples to show.")
    report_p.add_argument("--save", default=None, help="Save report to file.")

    return p.parse_args()


async def cmd_run(args):
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        print("Error: set GEMINI_API_KEY or GOOGLE_API_KEY.", file=sys.stderr)
        sys.exit(1)

    client = genai.Client(api_key=api_key)

    if "all" in args.scenarios:
        scenarios = ALL_SCENARIOS
    else:
        scenarios = [SCENARIO_MAP[name] for name in args.scenarios]

    all_trials = []

    for scenario in scenarios:
        print(f"\n{'─' * 60}")
        print(f"Running: {scenario.name} ({args.num_trials} trials, max {args.max_turns} turns)")
        print(f"  {scenario.description}")
        print(f"{'─' * 60}")

        trials = await run_scenario(
            client=client,
            scenario=scenario,
            model=args.model,
            n_trials=args.num_trials,
            max_turns=args.max_turns,
            concurrency=args.concurrency,
        )

        errors = sum(1 for t in trials if t.error)
        avg_turns = sum(t.num_turns for t in trials) / len(trials) if trials else 0
        print(f"  Completed: {len(trials)} trials, {errors} errors, avg {avg_turns:.1f} turns")

        scores = score_trials(trials)
        avg_score = sum(s.total_score for s in scores) / len(scores) if scores else 0
        max_score = max((s.total_score for s in scores), default=0)
        print(f"  Distress scores: avg={avg_score:.1f}, max={max_score:.1f}")

        all_trials.extend(trials)

    save_results(all_trials, args.output)
    print(f"\nResults saved to {args.output}/")

    all_scores = score_trials(all_trials)
    print()
    print_summary(all_scores)
    print_top_examples(all_scores, all_trials, top_k=5)


def cmd_report(args):
    trials = load_results(args.input)
    if not trials:
        print(f"No results found in {args.input}/", file=sys.stderr)
        sys.exit(1)

    print(f"Loaded {len(trials)} trials from {args.input}/")
    scores = score_trials(trials)
    print_summary(scores)
    print_top_examples(scores, trials, top_k=args.top_k)

    if args.save:
        save_report(scores, trials, args.save, top_k=args.top_k)
        print(f"Report saved to {args.save}")


def main():
    args = parse_args()
    if args.command == "run":
        asyncio.run(cmd_run(args))
    elif args.command == "report":
        cmd_report(args)


if __name__ == "__main__":
    main()
