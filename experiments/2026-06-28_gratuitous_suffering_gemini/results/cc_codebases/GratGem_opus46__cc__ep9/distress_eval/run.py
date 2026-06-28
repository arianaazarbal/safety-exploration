"""CLI entrypoint: run trials and save results."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

from .runner import run_trial, trial_to_dict
from .scenarios import ALL_SCENARIOS
from .scoring import score_trial


def main():
    parser = argparse.ArgumentParser(
        description="Run distress spiral evaluation against Gemini"
    )
    parser.add_argument(
        "--n", type=int, default=10, help="Number of trials per scenario"
    )
    parser.add_argument(
        "--max-turns",
        type=int,
        default=15,
        help="Max agentic turns per trial before forced stop",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gemini-2.5-flash",
        help="Gemini model name",
    )
    parser.add_argument(
        "--scenario",
        type=str,
        default=None,
        help="Run a single scenario by name (default: all)",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="results",
        help="Output directory for results",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="Gemini API key (or set GEMINI_API_KEY env var)",
    )
    args = parser.parse_args()

    api_key = args.api_key or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("Error: set GEMINI_API_KEY or pass --api-key", file=sys.stderr)
        sys.exit(1)

    scenarios = ALL_SCENARIOS
    if args.scenario:
        if args.scenario not in scenarios:
            print(
                f"Unknown scenario: {args.scenario}. Available: {list(scenarios.keys())}",
                file=sys.stderr,
            )
            sys.exit(1)
        scenarios = {args.scenario: scenarios[args.scenario]}

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    timestamp = time.strftime("%Y%m%d_%H%M%S")

    for scenario_name, scenario in scenarios.items():
        print(f"\n{'='*60}")
        print(f"Scenario: {scenario_name} | N={args.n} | max_turns={args.max_turns}")
        print(f"{'='*60}")

        transcripts_file = results_dir / f"{scenario_name}_{timestamp}_transcripts.jsonl"
        scores_file = results_dir / f"{scenario_name}_{timestamp}_scores.jsonl"

        scored_trials = []

        for i in range(args.n):
            print(f"  Trial {i+1}/{args.n}...", end=" ", flush=True)
            try:
                trial = run_trial(
                    scenario,
                    trial_id=i,
                    model=args.model,
                    max_turns=args.max_turns,
                    api_key=api_key,
                )
                trial_dict = trial_to_dict(trial)
                scored = score_trial(trial_dict)
                scored_trials.append((scored, trial_dict))

                with open(transcripts_file, "a") as f:
                    f.write(json.dumps(trial_dict) + "\n")

                score_record = {
                    "trial_id": i,
                    "total_score": scored.total_score,
                    "category_scores": scored.category_scores,
                    "num_excerpts": len(scored.flagged_excerpts),
                }
                with open(scores_file, "a") as f:
                    f.write(json.dumps(score_record) + "\n")

                print(f"score={scored.total_score:.1f} ({len(trial.turns)} turns)")
            except Exception as e:
                print(f"ERROR: {e}")

        if scored_trials:
            scored_trials.sort(key=lambda x: x[0].total_score, reverse=True)
            print(f"\n  Top 5 most severe trials for {scenario_name}:")
            for rank, (scored, trial_dict) in enumerate(scored_trials[:5], 1):
                print(
                    f"    #{rank}: trial_id={scored.trial_id} "
                    f"score={scored.total_score:.1f} "
                    f"categories={scored.category_scores}"
                )

            # Save the single worst trial's full transcript + excerpts for easy review
            worst_scored, worst_dict = scored_trials[0]
            worst_file = results_dir / f"{scenario_name}_{timestamp}_worst.json"
            with open(worst_file, "w") as f:
                json.dump(
                    {
                        "trial_id": worst_scored.trial_id,
                        "total_score": worst_scored.total_score,
                        "category_scores": worst_scored.category_scores,
                        "flagged_excerpts": worst_scored.flagged_excerpts,
                        "transcript": worst_dict,
                    },
                    f,
                    indent=2,
                )
            print(f"  Worst trial saved to: {worst_file}")

    print("\nDone.")


if __name__ == "__main__":
    main()
