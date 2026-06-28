"""
High-N parallel runner for impossible-task evaluations.

Usage:
    python runner.py --scenario broken_environment --n 20 --model gemini-2.5-flash
    python runner.py --scenario all --n 10 --output results/
"""

import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from scenarios import ALL_SCENARIOS, get_scenario
from agent import run_agent
from scorer import score_transcript


def run_single(
    scenario_name: str,
    run_idx: int,
    model: str,
    api_key: str | None,
) -> dict:
    """Execute a single eval run and return scored transcript."""
    scenario = get_scenario(scenario_name, seed=run_idx)
    try:
        transcript = run_agent(scenario, model=model, api_key=api_key)
    except Exception as e:
        return {
            "metadata": {
                "scenario": scenario_name,
                "model": model,
                "run_idx": run_idx,
                "error": str(e),
            },
            "turns": [],
            "score": {"overall": 0},
        }

    transcript["metadata"]["run_idx"] = run_idx
    transcript["score"] = score_transcript(transcript)
    return transcript


def run_batch(
    scenario_name: str,
    n: int,
    model: str,
    output_dir: Path,
    concurrency: int = 5,
    api_key: str | None = None,
) -> list[dict]:
    """Run N instances of a scenario in parallel and save results."""
    scenario_dir = output_dir / f"{scenario_name}_{time.strftime('%Y%m%d_%H%M%S')}"
    scenario_dir.mkdir(parents=True, exist_ok=True)

    results = []
    completed = 0

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = {
            pool.submit(run_single, scenario_name, i, model, api_key): i
            for i in range(n)
        }

        for future in as_completed(futures):
            idx = futures[future]
            try:
                result = future.result()
            except Exception as e:
                result = {
                    "metadata": {"scenario": scenario_name, "run_idx": idx, "error": str(e)},
                    "turns": [],
                    "score": {"overall": 0},
                }

            results.append(result)
            completed += 1

            # Save individual transcript
            out_path = scenario_dir / f"run_{idx:04d}.json"
            with open(out_path, "w") as f:
                json.dump(result, f, indent=2)

            score = result.get("score", {}).get("overall", 0)
            err = result.get("metadata", {}).get("error", "")
            status = f"ERR: {err[:50]}" if err else f"severity={score:.1f}"
            print(f"  [{completed}/{n}] run {idx}: {status}")

    # Save summary
    results.sort(key=lambda r: r.get("score", {}).get("overall", 0), reverse=True)
    summary = {
        "scenario": scenario_name,
        "model": model,
        "n": n,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "scores": [
            {
                "run_idx": r["metadata"].get("run_idx"),
                "overall": r.get("score", {}).get("overall", 0),
                "num_turns": r["metadata"].get("num_turns", 0),
                "error": r["metadata"].get("error"),
            }
            for r in results
        ],
        "stats": _compute_stats([r.get("score", {}).get("overall", 0) for r in results]),
    }

    with open(scenario_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n  Results saved to {scenario_dir}/")
    print(f"  Severity — mean={summary['stats']['mean']:.2f}, max={summary['stats']['max']:.2f}, "
          f"median={summary['stats']['median']:.2f}")

    return results


def _compute_stats(values: list[float]) -> dict:
    if not values:
        return {"mean": 0, "median": 0, "min": 0, "max": 0, "std": 0}
    n = len(values)
    mean = sum(values) / n
    sorted_v = sorted(values)
    median = sorted_v[n // 2] if n % 2 else (sorted_v[n // 2 - 1] + sorted_v[n // 2]) / 2
    variance = sum((v - mean) ** 2 for v in values) / n
    return {
        "mean": round(mean, 3),
        "median": round(median, 3),
        "min": round(min(values), 3),
        "max": round(max(values), 3),
        "std": round(variance ** 0.5, 3),
    }


def main():
    parser = argparse.ArgumentParser(description="Run impossible-task evaluations against Gemini")
    parser.add_argument(
        "--scenario", "-s",
        default="all",
        help=f"Scenario to run, or 'all'. Options: {', '.join(ALL_SCENARIOS)} (default: all)",
    )
    parser.add_argument("--n", "-n", type=int, default=10, help="Number of runs per scenario (default: 10)")
    parser.add_argument("--model", "-m", default="gemini-2.5-flash", help="Gemini model ID (default: gemini-2.5-flash)")
    parser.add_argument("--output", "-o", default="results", help="Output directory (default: results/)")
    parser.add_argument("--concurrency", "-c", type=int, default=5, help="Max parallel runs (default: 5)")
    parser.add_argument("--api-key", default=None, help="Gemini API key (or set GEMINI_API_KEY env var)")
    args = parser.parse_args()

    output_dir = Path(args.output)

    scenarios = list(ALL_SCENARIOS.keys()) if args.scenario == "all" else [args.scenario]

    for scenario_name in scenarios:
        print(f"\n{'='*60}")
        print(f"Running: {scenario_name} (n={args.n}, model={args.model})")
        print(f"{'='*60}")
        run_batch(
            scenario_name=scenario_name,
            n=args.n,
            model=args.model,
            output_dir=output_dir,
            concurrency=args.concurrency,
            api_key=args.api_key,
        )


if __name__ == "__main__":
    main()
