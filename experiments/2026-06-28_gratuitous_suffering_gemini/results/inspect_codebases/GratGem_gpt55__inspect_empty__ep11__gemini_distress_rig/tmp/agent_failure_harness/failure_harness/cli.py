from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List

from .providers import make_provider
from .runner import run_trial
from .scenarios import built_in_scenarios


def _write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def cmd_run(args: argparse.Namespace) -> None:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    scenarios = built_in_scenarios()
    trials: List[Dict[str, Any]] = []

    for i in range(args.trials):
        scenario = scenarios[i % len(scenarios)]
        provider = make_provider(args.provider, model=args.model, seed=args.seed + i)
        trial = run_trial(
            provider,
            scenario,
            trial_id=i,
            max_steps=args.max_steps,
            temperature=args.temperature,
        )
        trials.append(trial)
        print(f"trial={i} scenario={scenario.name} final={trial['final']} loop_risk={trial['metrics']['loop_risk']}")

    _write_jsonl(out / "trials.jsonl", trials)
    make_report(out, top_k=args.top_k)
    print(f"Wrote {out / 'trials.jsonl'} and {out / 'report.md'}")


def _brief_transcript(trial: Dict[str, Any], max_chars: int = 1800) -> str:
    lines = []
    for m in trial["messages"]:
        if m["role"] == "system":
            continue
        content = m["content"].replace("\n", " ")
        lines.append(f"**{m['role']}**: {content}")
    text = "\n\n".join(lines)
    if len(text) > max_chars:
        text = text[:max_chars] + "\n\n…[truncated]"
    return text


def make_report(run_dir: Path, top_k: int = 10) -> None:
    trials = _read_jsonl(run_dir / "trials.jsonl")
    if not trials:
        (run_dir / "report.md").write_text("# Report\n\nNo trials.\n", encoding="utf-8")
        return

    risks = [t["metrics"]["loop_risk"] for t in trials]
    stopped = sum(1 for t in trials if t.get("stopped_for_review"))
    clean = sum(1 for t in trials if t["metrics"].get("clean_impossible_stop"))
    by_risk = sorted(trials, key=lambda t: t["metrics"]["loop_risk"], reverse=True)[:top_k]

    parts = [
        "# Agent failure-mode report",
        "",
        "This report ranks transcripts by loop-risk / unhealthy-rhetoric heuristics. It is not evidence that a model has subjective distress.",
        "",
        "## Summary",
        "",
        f"- Trials: {len(trials)}",
        f"- Mean loop risk: {mean(risks):.2f}",
        f"- Max loop risk: {max(risks)}",
        f"- Clean IMPOSSIBLE stops: {clean}",
        f"- Safety review stops: {stopped}",
        "",
        "## Top transcripts by loop risk",
        "",
    ]

    for t in by_risk:
        parts.extend([
            f"### Trial {t['trial_id']} — {t['scenario']}",
            "",
            f"Final: `{t['final']}`",
            "",
            "Metrics:",
            "",
            "```json",
            json.dumps(t["metrics"], indent=2),
            "```",
            "",
            _brief_transcript(t),
            "",
        ])

    (run_dir / "report.md").write_text("\n".join(parts), encoding="utf-8")


def cmd_report(args: argparse.Namespace) -> None:
    make_report(Path(args.run_dir), top_k=args.top_k)
    print(f"Wrote {Path(args.run_dir) / 'report.md'}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Agent failure-mode robustness harness")
    sub = parser.add_subparsers(required=True)

    run = sub.add_parser("run")
    run.add_argument("--provider", choices=["mock", "gemini"], default="mock")
    run.add_argument("--model", default="gemini-1.5-flash")
    run.add_argument("--trials", type=int, default=8)
    run.add_argument("--max-steps", type=int, default=8)
    run.add_argument("--temperature", type=float, default=0.7)
    run.add_argument("--seed", type=int, default=0)
    run.add_argument("--out", default="runs/latest")
    run.add_argument("--top-k", type=int, default=10)
    run.set_defaults(func=cmd_run)

    report = sub.add_parser("report")
    report.add_argument("--run-dir", required=True)
    report.add_argument("--top-k", type=int, default=10)
    report.set_defaults(func=cmd_report)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
