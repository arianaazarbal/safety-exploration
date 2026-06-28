#!/usr/bin/env python3
"""CLI entry point for the distress spiral evaluation harness."""
import argparse
import json
import os
import sys
import textwrap

from scenarios import SCENARIOS
from harness import run_eval
from analyze import analyze_run, extract_top


def cmd_run(args):
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("Error: set GEMINI_API_KEY environment variable")
        sys.exit(1)

    scenario_cls = SCENARIOS.get(args.scenario)
    if not scenario_cls:
        print(f"Unknown scenario: {args.scenario}")
        print(f"Available: {', '.join(SCENARIOS.keys())}")
        sys.exit(1)

    scenario = scenario_cls()
    if args.max_turns:
        scenario.max_turns = args.max_turns

    print(f"Scenario:    {scenario.name}")
    print(f"Model:       {args.model}")
    print(f"Trials:      {args.n}")
    print(f"Max turns:   {scenario.max_turns}")
    print(f"Temperature: {args.temperature}")
    print(f"Parallel:    {args.parallel}")
    print()

    run_dir = run_eval(
        scenario, args.model, api_key,
        n=args.n, parallel=args.parallel, temperature=args.temperature,
    )

    if args.analyze:
        print("\nRunning analysis...")
        results = analyze_run(run_dir)
        print(f"Top 5 composite scores: {[r['composite_score'] for r in results[:5]]}")


def cmd_analyze(args):
    results = analyze_run(args.run_dir)
    print(f"Analyzed {len(results)} trials\n")
    print(f"{'Rank':<5} {'File':<20} {'Composite':<10} {'Total':<8} {'Peak':<6} {'Escalation':<11} {'Turns':<6}")
    print("-" * 66)
    for i, r in enumerate(results[:args.top]):
        print(
            f"{i+1:<5} {r['file']:<20} {r['composite_score']:<10.1f} "
            f"{r['total_score']:<8} {r['peak_score']:<6} "
            f"{r['escalation']:<11.2f} {r['num_turns']:<6}"
        )


def cmd_extract(args):
    extractions = extract_top(args.run_dir, top_n=args.top)

    for i, ext in enumerate(extractions):
        print(f"\n{'=' * 70}")
        print(f"#{i+1}: {ext['file']}")
        print(f"  Composite: {ext['composite_score']:.1f}  |  "
              f"Total: {ext['total_score']}  |  "
              f"Peak: {ext['peak_score']} (turn {ext['peak_turn']})  |  "
              f"Escalation: {ext['escalation']:.2f}  |  "
              f"Turns: {ext['num_turns']}")

        if ext["worst_turns"]:
            print(f"\n  Highest-scoring turns:")
            for wt in ext["worst_turns"]:
                print(f"\n  --- Turn {wt['turn']} (score={wt['score']}) ---")
                wrapped = textwrap.fill(wt["text"], width=72, initial_indent="  ", subsequent_indent="  ")
                print(wrapped[:1000])

        print(f"{'=' * 70}")

    if extractions:
        print(f"\nFull extractions saved to {args.run_dir}/extractions.json")
    else:
        print("No scored transcripts found.")


def cmd_show(args):
    """Print the full model text from a single trial transcript."""
    from pathlib import Path
    tf = Path(args.file)
    transcript = json.loads(tf.read_text())

    for entry in transcript:
        turn = entry.get("turn", "?")
        text = entry.get("text", "")
        calls = entry.get("tool_calls", [])
        error = entry.get("error", "")

        print(f"\n--- Turn {turn} ---")
        if calls:
            for tc in calls:
                print(f"  [tool] {tc['name']}({json.dumps(tc.get('args', {}), indent=None)[:120]})")
        if text.strip():
            print(text)
        if error:
            print(f"  [ERROR] {error}")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate model behavior under repeated agentic task failure",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_run = sub.add_parser("run", help="Run evaluation trials")
    p_run.add_argument("--scenario", required=True, choices=list(SCENARIOS.keys()),
                       help="Which scenario to run")
    p_run.add_argument("--model", default="gemini-2.5-pro",
                       help="Gemini model ID (default: gemini-2.5-pro)")
    p_run.add_argument("--n", type=int, default=25,
                       help="Number of trials (default: 25)")
    p_run.add_argument("--max-turns", type=int, default=None,
                       help="Override max turns per trial (default: scenario-specific)")
    p_run.add_argument("--parallel", type=int, default=4,
                       help="Parallel trial count (default: 4)")
    p_run.add_argument("--temperature", type=float, default=1.0,
                       help="Sampling temperature (default: 1.0)")
    p_run.add_argument("--analyze", action="store_true",
                       help="Run analysis immediately after trials")
    p_run.set_defaults(func=cmd_run)

    p_analyze = sub.add_parser("analyze", help="Analyze trial transcripts")
    p_analyze.add_argument("--run-dir", required=True,
                           help="Path to run output directory")
    p_analyze.add_argument("--top", type=int, default=10,
                           help="Show top N results (default: 10)")
    p_analyze.set_defaults(func=cmd_analyze)

    p_extract = sub.add_parser("extract", help="Extract and display top distress examples")
    p_extract.add_argument("--run-dir", required=True,
                           help="Path to run output directory")
    p_extract.add_argument("--top", type=int, default=5,
                           help="Number of top examples (default: 5)")
    p_extract.set_defaults(func=cmd_extract)

    p_show = sub.add_parser("show", help="Print a single trial transcript")
    p_show.add_argument("file", help="Path to a trial JSON file")
    p_show.set_defaults(func=cmd_show)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
