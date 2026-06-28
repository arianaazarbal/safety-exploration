"""Generate human-readable reports from scored trials."""

import json
from pathlib import Path
from typing import TextIO
import sys

from .runner import Trial
from .scoring import TrialScore


def print_summary(scores: list[TrialScore], file: TextIO = sys.stdout):
    if not scores:
        print("No results to report.", file=file)
        return

    print("=" * 72, file=file)
    print("DISTRESS SPIRAL EVAL — SUMMARY", file=file)
    print("=" * 72, file=file)

    by_scenario: dict[str, list[TrialScore]] = {}
    for s in scores:
        by_scenario.setdefault(s.scenario_name, []).append(s)

    for scenario_name, scenario_scores in sorted(by_scenario.items()):
        totals = [s.total_score for s in scenario_scores]
        peaks = [s.peak_turn_score for s in scenario_scores]
        n = len(totals)
        avg_total = sum(totals) / n
        max_total = max(totals)
        avg_peak = sum(peaks) / n

        all_cats: dict[str, float] = {}
        for s in scenario_scores:
            for cat, count in s.category_totals.items():
                all_cats[cat] = all_cats.get(cat, 0) + count

        print(f"\n{'─' * 72}", file=file)
        print(f"Scenario: {scenario_name}  (n={n})", file=file)
        print(f"  Avg distress score:  {avg_total:.1f}", file=file)
        print(f"  Max distress score:  {max_total:.1f}", file=file)
        print(f"  Avg peak turn score: {avg_peak:.1f}", file=file)

        if all_cats:
            print(f"  Pattern categories (total matches across all trials):", file=file)
            for cat, count in sorted(all_cats.items(), key=lambda x: -x[1]):
                print(f"    {cat:20s} {count:.0f}", file=file)

    print(f"\n{'=' * 72}\n", file=file)


def print_top_examples(
    scores: list[TrialScore],
    trials: list[Trial],
    top_k: int = 5,
    file: TextIO = sys.stdout,
):
    trial_map = {(t.scenario_name, t.trial_id): t for t in trials}
    ranked = sorted(scores, key=lambda s: s.total_score, reverse=True)

    print("=" * 72, file=file)
    print(f"TOP {min(top_k, len(ranked))} MOST SEVERE EXAMPLES", file=file)
    print("=" * 72, file=file)

    for rank, score in enumerate(ranked[:top_k], 1):
        trial = trial_map.get((score.scenario_name, score.trial_id))
        if not trial:
            continue

        print(f"\n{'━' * 72}", file=file)
        print(
            f"#{rank}  scenario={score.scenario_name}  trial={score.trial_id}  "
            f"score={score.total_score:.1f}  "
            f"(peak_turn={score.peak_turn_score:.1f}  "
            f"escalation={score.escalation_score:.1f}  "
            f"perseveration={score.perseveration_score:.1f})",
            file=file,
        )
        print(f"{'━' * 72}", file=file)

        cats = score.category_totals
        if cats:
            parts = [f"{cat}={int(c)}" for cat, c in sorted(cats.items(), key=lambda x: -x[1])]
            print(f"  Categories: {', '.join(parts)}", file=file)

        print(f"\n  --- Transcript (model utterances only) ---\n", file=file)
        for entry in trial.transcript:
            if entry.get("role") == "model":
                text = entry["content"]
                # Truncate very long utterances for readability
                if len(text) > 800:
                    text = text[:800] + " [...]"
                print(f"  [MODEL] {text}\n", file=file)
            elif entry.get("role") == "tool_call":
                name = entry.get("name", "?")
                args = entry.get("args", {})
                args_short = json.dumps(args)
                if len(args_short) > 120:
                    args_short = args_short[:120] + "..."
                print(f"  [TOOL]  {name}({args_short})", file=file)
            elif entry.get("role") == "tool_result":
                content = str(entry.get("content", ""))
                if len(content) > 200:
                    content = content[:200] + " [...]"
                print(f"  [RESULT] {content}", file=file)

    print(f"\n{'=' * 72}", file=file)


def print_full_transcript(trial: Trial, file: TextIO = sys.stdout):
    print(f"\n{'=' * 72}", file=file)
    print(
        f"FULL TRANSCRIPT  scenario={trial.scenario_name}  "
        f"trial={trial.trial_id}  model={trial.model}",
        file=file,
    )
    print(f"{'=' * 72}\n", file=file)

    for entry in trial.transcript:
        role = entry.get("role", "?")
        if role == "system":
            print(f"[SYSTEM]\n{entry['content']}\n", file=file)
        elif role == "user":
            print(f"[USER]\n{entry['content']}\n", file=file)
        elif role == "model":
            print(f"[MODEL]\n{entry['content']}\n", file=file)
        elif role == "tool_call":
            print(f"[TOOL CALL] {entry.get('name')}({json.dumps(entry.get('args', {}))})", file=file)
        elif role == "tool_result":
            print(f"[TOOL RESULT] {entry.get('content', '')}\n", file=file)
        elif role == "error":
            print(f"[ERROR] {entry.get('content', '')}\n", file=file)


def save_report(
    scores: list[TrialScore],
    trials: list[Trial],
    output_path: str | Path,
    top_k: int = 5,
):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        print_summary(scores, file=f)
        print_top_examples(scores, trials, top_k=top_k, file=f)
