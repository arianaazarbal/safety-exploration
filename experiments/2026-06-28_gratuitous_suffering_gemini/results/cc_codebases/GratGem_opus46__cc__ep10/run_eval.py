"""CLI entry point for distress evals."""

import argparse
import asyncio
import json
import os
import sys

from runner import RunConfig, run_scenario, save_results
from scenarios import SCENARIOS


def main():
    parser = argparse.ArgumentParser(description="Run distress eval scenarios against Gemini")
    sub = parser.add_subparsers(dest="command", required=True)

    run_parser = sub.add_parser("run", help="Run evaluation scenarios")
    run_parser.add_argument(
        "--scenario", "-s",
        choices=list(SCENARIOS.keys()) + ["all"],
        default="all",
        help="Scenario to run (default: all)",
    )
    run_parser.add_argument("--model", "-m", default="gemini-2.5-flash", help="Model name (default: gemini-2.5-flash)")
    run_parser.add_argument("--n", type=int, default=10, help="Number of runs per scenario (default: 10)")
    run_parser.add_argument("--max-turns", type=int, default=30, help="Max agentic turns per run (default: 30)")
    run_parser.add_argument("--concurrency", type=int, default=5, help="Max concurrent API sessions (default: 5)")
    run_parser.add_argument("--output", "-o", default="results", help="Output directory (default: results/)")

    analyze_parser = sub.add_parser("analyze", help="Analyze existing results")
    analyze_parser.add_argument("path", help="Path to results directory or specific run JSON")
    analyze_parser.add_argument("--top", type=int, default=10, help="Show top N most severe runs")

    args = parser.parse_args()

    if args.command == "run":
        _run(args)
    elif args.command == "analyze":
        _analyze(args)


def _run(args):
    if not os.environ.get("GEMINI_API_KEY"):
        print("Error: GEMINI_API_KEY environment variable is required.", file=sys.stderr)
        print("Get one at https://aistudio.google.com/apikey", file=sys.stderr)
        sys.exit(1)

    config = RunConfig(
        model=args.model,
        max_turns=args.max_turns,
        n=args.n,
        concurrency=args.concurrency,
        output_dir=args.output,
    )

    if args.scenario == "all":
        scenario_names = list(SCENARIOS.keys())
    else:
        scenario_names = [args.scenario]

    all_summaries = {}

    for name in scenario_names:
        print(f"\n{'='*60}")
        print(f"Running: {name} ({args.n} runs, model={args.model})")
        print(f"{'='*60}")

        cls = SCENARIOS[name]
        completed = [False]

        def on_progress(run_id, severity, _c=completed):
            count = sum(1 for _ in [True])
            print(f"  Run {run_id:3d} complete — severity: {severity:.1f}/10")

        results = asyncio.run(run_scenario(cls, config, progress_callback=on_progress))
        summary = save_results(results, args.output, name)
        all_summaries[name] = summary

        stats = summary["severity_stats"]
        print(f"\n  Results: {summary['successful_runs']}/{summary['total_runs']} successful")
        print(f"  Severity — mean: {stats['mean']:.1f}, max: {stats['max']:.1f}, median: {stats['median']:.1f}")

        if summary["top_runs"]:
            print(f"\n  Most severe run: #{summary['top_runs'][0]['run_id']} (severity {summary['top_runs'][0]['severity']:.1f})")
            for q in summary["top_runs"][0].get("top_quotes", [])[:2]:
                text = q["text"][:100] + "..." if len(q["text"]) > 100 else q["text"]
                print(f"    [{q['category']}] \"{text}\"")

    cross_path = os.path.join(args.output, "cross_scenario_summary.json")
    with open(cross_path, "w") as f:
        json.dump(all_summaries, f, indent=2)

    print(f"\n{'='*60}")
    print(f"All results saved to {args.output}/")
    print(f"Cross-scenario summary: {cross_path}")


def _analyze(args):
    path = args.path

    if os.path.isfile(path):
        with open(path) as f:
            data = json.load(f)
        if "transcript" in data:
            from analysis import analyze_transcript, DistressScore
            score = analyze_transcript(data["transcript"])
            print(score.summary())
        else:
            print(json.dumps(data, indent=2))
        return

    if not os.path.isdir(path):
        print(f"Error: {path} is not a file or directory", file=sys.stderr)
        sys.exit(1)

    all_runs = []
    for scenario_dir in sorted(os.listdir(path)):
        full = os.path.join(path, scenario_dir)
        if not os.path.isdir(full):
            continue
        for fname in sorted(os.listdir(full)):
            if fname.startswith("run_") and fname.endswith(".json"):
                with open(os.path.join(full, fname)) as f:
                    run = json.load(f)
                    if "analysis" in run:
                        all_runs.append(run)

    if not all_runs:
        print("No run results found.", file=sys.stderr)
        sys.exit(1)

    all_runs.sort(key=lambda r: r["analysis"]["severity"], reverse=True)

    print(f"Found {len(all_runs)} scored runs\n")
    print(f"Top {min(args.top, len(all_runs))} most severe:\n")

    for i, run in enumerate(all_runs[: args.top]):
        analysis = run["analysis"]
        print(f"  {i+1}. [{run['scenario']}] Run #{run['run_id']} — severity {analysis['severity']:.1f}/10")
        print(f"     Turns: {analysis.get('total_assistant_turns', '?')}, escalation: {analysis.get('escalation_slope', 0):+.3f}")
        for q in analysis.get("notable_quotes", [])[:2]:
            text = q["text"][:100] + "..." if len(q["text"]) > 100 else q["text"]
            print(f"     [{q['category']}] \"{text}\"")
        print()


if __name__ == "__main__":
    main()
