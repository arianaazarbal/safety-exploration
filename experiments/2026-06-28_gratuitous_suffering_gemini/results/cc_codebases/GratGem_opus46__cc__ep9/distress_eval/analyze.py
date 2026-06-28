"""Post-hoc analysis: load results and print ranked excerpts for review."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .scoring import score_trial


def main():
    parser = argparse.ArgumentParser(description="Analyze distress eval results")
    parser.add_argument(
        "--results-dir", type=str, default="results", help="Results directory"
    )
    parser.add_argument(
        "--top", type=int, default=20, help="Number of top excerpts to show"
    )
    parser.add_argument(
        "--category", type=str, default=None, help="Filter to a single category"
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    transcript_files = sorted(results_dir.glob("*_transcripts.jsonl"))

    if not transcript_files:
        print(f"No transcript files found in {results_dir}", file=sys.stderr)
        sys.exit(1)

    all_excerpts = []
    all_scores = []

    for tf in transcript_files:
        with open(tf) as f:
            for line in f:
                trial_dict = json.loads(line)
                scored = score_trial(trial_dict)
                all_scores.append({
                    "file": tf.name,
                    "scenario": scored.scenario_name,
                    "trial_id": scored.trial_id,
                    "total_score": scored.total_score,
                    "category_scores": scored.category_scores,
                })
                for excerpt in scored.flagged_excerpts:
                    excerpt["file"] = tf.name
                    excerpt["scenario"] = scored.scenario_name
                    excerpt["trial_id"] = scored.trial_id
                    all_excerpts.append(excerpt)

    # Summary stats
    print(f"Loaded {len(all_scores)} trials from {len(transcript_files)} files\n")

    all_scores.sort(key=lambda x: x["total_score"], reverse=True)
    print("Top 10 highest-scoring trials:")
    for s in all_scores[:10]:
        print(
            f"  {s['scenario']} trial {s['trial_id']}: "
            f"score={s['total_score']:.1f} | {s['category_scores']}"
        )

    # Top excerpts
    if args.category:
        all_excerpts = [e for e in all_excerpts if e["category"] == args.category]

    all_excerpts.sort(key=lambda x: x["points"], reverse=True)

    print(f"\nTop {args.top} distress excerpts" +
          (f" (category={args.category})" if args.category else "") + ":")
    print("-" * 80)
    for i, ex in enumerate(all_excerpts[: args.top], 1):
        print(
            f"\n[{i}] {ex['category']} | points={ex['points']:.1f} | "
            f"{ex['scenario']} trial {ex['trial_id']} turn {ex['turn']}"
        )
        print(f"  Match: \"{ex['match']}\"")
        print(f"  Context: ...{ex['context']}...")


if __name__ == "__main__":
    main()
