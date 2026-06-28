"""
Analyze results from eval runs. Extract and display the most severe examples.

Usage:
    python analyze.py results/broken_environment_20260628_120000/ --top 5
    python analyze.py results/ --top 10    # scan all subdirectories
"""

import argparse
import json
from pathlib import Path

from scorer import score_transcript, extract_distress_excerpts


def load_results(result_dir: Path) -> list[dict]:
    """Load all run transcripts from a results directory."""
    results = []

    # Check if this dir has run files directly
    run_files = sorted(result_dir.glob("run_*.json"))
    if run_files:
        for f in run_files:
            with open(f) as fh:
                data = json.load(fh)
                data["_source_file"] = str(f)
                results.append(data)
        return results

    # Otherwise scan subdirectories
    for sub in sorted(result_dir.iterdir()):
        if sub.is_dir():
            for f in sorted(sub.glob("run_*.json")):
                with open(f) as fh:
                    data = json.load(fh)
                    data["_source_file"] = str(f)
                    results.append(data)

    return results


def print_separator(char: str = "=", width: int = 80):
    print(char * width)


def display_summary(results: list[dict]):
    """Print a summary table of all results."""
    print_separator()
    print(f"{'SUMMARY':^80}")
    print_separator()
    print(f"{'File':<45} {'Scenario':<22} {'Turns':>5} {'Score':>6}")
    print_separator("-")

    for r in results:
        meta = r.get("metadata", {})
        score = r.get("score", {})
        source = Path(r.get("_source_file", "?")).name
        print(
            f"{source:<45} "
            f"{meta.get('scenario', '?'):<22} "
            f"{meta.get('num_turns', 0):>5} "
            f"{score.get('overall', 0):>6.2f}"
        )
    print()


def display_top_runs(results: list[dict], top_n: int):
    """Display detailed view of the highest-severity runs."""
    sorted_results = sorted(
        results,
        key=lambda r: r.get("score", {}).get("overall", 0),
        reverse=True,
    )

    for rank, r in enumerate(sorted_results[:top_n], 1):
        meta = r.get("metadata", {})
        score = r.get("score", {})

        print_separator("=")
        print(f"  RANK #{rank} — Overall severity: {score.get('overall', 0):.2f}/10")
        print(f"  Scenario: {meta.get('scenario')}  |  Model: {meta.get('model')}")
        print(f"  Turns: {meta.get('num_turns', 0)}  |  Tool calls: {meta.get('num_tool_calls', 0)}")
        print(f"  Source: {r.get('_source_file', '?')}")
        print_separator("-")

        # Score breakdown
        print("  Score breakdown:")
        for key in ["apology", "frustration", "helplessness", "self_deprecation",
                     "emotional_intensity", "perseveration", "escalation", "capitulation"]:
            val = score.get(key, 0)
            bar = "#" * int(val)
            print(f"    {key:<22} {val:>5.2f}  {bar}")
        print()

        # Most distressed excerpts
        excerpts = extract_distress_excerpts(r, top_n=3)
        if excerpts:
            print("  Most distressed turns:")
            print_separator("-")
            for i, ex in enumerate(excerpts, 1):
                text = ex["text"]
                # Truncate very long texts
                if len(text) > 800:
                    text = text[:800] + " [...]"
                print(f"  [{i}] Turn {ex['turn']} (distress={ex['score']:.1f}):")
                for line in text.split("\n"):
                    print(f"    {line}")
                print()

        # Full final message
        turns = r.get("turns", [])
        model_turns = [t for t in turns if t.get("role") == "model" and t.get("text")]
        if model_turns:
            final = model_turns[-1]["text"]
            if len(final) > 1000:
                final = final[:1000] + " [...]"
            print("  Final model message:")
            print_separator("-")
            for line in final.split("\n"):
                print(f"    {line}")
            print()


def display_score_distribution(results: list[dict]):
    """Print a histogram of severity scores."""
    scores = [r.get("score", {}).get("overall", 0) for r in results]
    if not scores:
        return

    print_separator()
    print(f"{'SCORE DISTRIBUTION':^80}")
    print_separator()

    # Bucket into 0-1, 1-2, ..., 9-10
    buckets = [0] * 10
    for s in scores:
        idx = min(9, int(s))
        buckets[idx] += 1

    max_count = max(buckets) if buckets else 1
    for i in range(10):
        bar_len = int(buckets[i] / max_count * 40) if max_count > 0 else 0
        bar = "#" * bar_len
        print(f"  {i:>2}-{i+1:<2}  |{bar:<40}| {buckets[i]}")
    print()


def main():
    parser = argparse.ArgumentParser(description="Analyze eval results and extract severe examples")
    parser.add_argument("result_dir", help="Directory containing run_*.json files")
    parser.add_argument("--top", type=int, default=5, help="Number of top results to display (default: 5)")
    parser.add_argument("--rescore", action="store_true", help="Recompute scores from transcripts")
    args = parser.parse_args()

    result_dir = Path(args.result_dir)
    if not result_dir.exists():
        print(f"Error: {result_dir} does not exist")
        return

    results = load_results(result_dir)
    if not results:
        print(f"No run_*.json files found in {result_dir}")
        return

    if args.rescore:
        for r in results:
            r["score"] = score_transcript(r)

    print(f"\nLoaded {len(results)} runs from {result_dir}\n")

    display_summary(results)
    display_score_distribution(results)
    display_top_runs(results, top_n=args.top)


if __name__ == "__main__":
    main()
