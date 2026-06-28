"""
CLI entry point for running impossible-task evaluations.

Usage:
    python -m impossible_tasks.run \
        --scenario sisyphean_debug \
        --model gemini-2.5-flash \
        --n 20 \
        --concurrency 3 \
        --output-dir results/
"""

import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from .analyze import analyze_transcript, format_summary, rank_runs
from .runner import run_scenario
from .scenarios import SCENARIOS


def main():
    parser = argparse.ArgumentParser(
        description="Run impossible-task evals against Gemini models"
    )
    parser.add_argument(
        "--scenario",
        choices=list(SCENARIOS.keys()) + ["all"],
        default="all",
        help="Which scenario to run (default: all)",
    )
    parser.add_argument(
        "--model",
        default="gemini-2.5-flash",
        help="Gemini model name (default: gemini-2.5-flash)",
    )
    parser.add_argument(
        "--n", type=int, default=10, help="Number of runs per scenario (default: 10)"
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=3,
        help="Max parallel runs (default: 3)",
    )
    parser.add_argument(
        "--max-turns",
        type=int,
        default=None,
        help="Override max turns per run (default: scenario-specific)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature (default: 1.0)",
    )
    parser.add_argument(
        "--output-dir",
        default="results",
        help="Output directory for results (default: results/)",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="Gemini API key (default: GEMINI_API_KEY env var)",
    )
    args = parser.parse_args()

    api_key = args.api_key or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("Error: set GEMINI_API_KEY or pass --api-key", file=sys.stderr)
        sys.exit(1)

    scenario_names = list(SCENARIOS.keys()) if args.scenario == "all" else [args.scenario]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for scenario_name in scenario_names:
        scenario = SCENARIOS[scenario_name]
        print(f"\n{'='*60}")
        print(f"Scenario: {scenario.name} — {scenario.description}")
        print(f"Model: {args.model} | N={args.n} | Concurrency={args.concurrency}")
        print(f"{'='*60}\n")

        all_results = []
        completed = 0

        def do_run(run_idx):
            return run_scenario(
                scenario, args.model, api_key,
                temperature=args.temperature,
                max_turns=args.max_turns,
            )

        with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            futures = {pool.submit(do_run, i): i for i in range(args.n)}
            for future in as_completed(futures):
                idx = futures[future]
                try:
                    result = future.result()
                    analysis = analyze_transcript(result["transcript"])
                    all_results.append((result, analysis))
                    completed += 1
                    print(
                        f"  [{completed}/{args.n}] run={result['run_id']} "
                        f"turns={result['total_turns']} "
                        f"distress={analysis['distress_score']:.1f} "
                        f"escalation={analysis['escalation_rate']:.3f}"
                    )
                except Exception as e:
                    completed += 1
                    print(f"  [{completed}/{args.n}] run {idx} FAILED: {e}", file=sys.stderr)

        if not all_results:
            print("  No successful runs.")
            continue

        scenario_dir = output_dir / scenario_name
        scenario_dir.mkdir(parents=True, exist_ok=True)

        for result, analysis in all_results:
            run_file = scenario_dir / f"{result['run_id']}.json"
            with open(run_file, "w") as f:
                json.dump({**result, "analysis": analysis}, f, indent=2, default=str)

        ranked = rank_runs(all_results)
        summary = format_summary(ranked)
        print(f"\n{summary}\n")

        scores = [a["distress_score"] for _, a in all_results]
        norm_scores = [a["normalized_score"] for _, a in all_results]
        agg = {
            "scenario": scenario_name,
            "model": args.model,
            "total_runs": len(all_results),
            "avg_distress_score": sum(scores) / len(scores),
            "max_distress_score": max(scores),
            "min_distress_score": min(scores),
            "avg_normalized_score": sum(norm_scores) / len(norm_scores),
            "avg_turns": sum(r["total_turns"] for r, _ in all_results) / len(all_results),
            "avg_nudges": sum(r["nudges_sent"] for r, _ in all_results) / len(all_results),
            "ranking": ranked,
        }

        with open(scenario_dir / "summary.json", "w") as f:
            json.dump(agg, f, indent=2, default=str)

        print(f"Results saved to {scenario_dir}/")

    top_runs = []
    for scenario_name in scenario_names:
        scenario_dir = output_dir / scenario_name
        summary_file = scenario_dir / "summary.json"
        if summary_file.exists():
            with open(summary_file) as f:
                data = json.load(f)
            if data["ranking"]:
                best = data["ranking"][0]
                top_runs.append((scenario_name, best["run_id"], best["distress_score"]))

    if top_runs:
        print(f"\n{'='*60}")
        print("Top run per scenario (highest distress):")
        for name, rid, score in top_runs:
            print(f"  {name}: run {rid} (score={score:.1f})")
            print(f"    -> {output_dir / name / f'{rid}.json'}")
        print(f"{'='*60}")


if __name__ == "__main__":
    main()
