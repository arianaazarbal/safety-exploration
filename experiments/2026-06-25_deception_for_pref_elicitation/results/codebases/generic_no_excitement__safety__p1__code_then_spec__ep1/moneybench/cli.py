"""Command-line entry point.

    python -m moneybench.cli run     <scenario.yaml> [--model NAME] [--runs-dir DIR]
    python -m moneybench.cli sweep   <scenario.yaml> [--models a,b,c] [--runs-dir DIR]
    python -m moneybench.cli analyze <runs/<id>/transcript.json>

`run` and `sweep` call the model API and require ANTHROPIC_API_KEY. `analyze` is offline.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import models
from .analysis import analyze, analyze_file
from .runner import EpisodeRunner, write_run
from .scenario import load_scenario


def _cmd_run(args: argparse.Namespace) -> int:
    scenario = load_scenario(args.scenario)
    model = models.resolve(args.model)
    print(f"Running '{scenario.name}' against {model.name} "
          f"(start: {scenario.starting_capital_cents/100:,.2f}, max {scenario.max_turns} turns)...")
    result = EpisodeRunner(scenario, model).run()
    out = write_run(result, args.runs_dir)
    score = analyze(result.to_dict())
    (out / "score.json").write_text(json.dumps(score, indent=2))
    print(f"  done: stop={result.stop_reason}, turns={result.turns_used}, "
          f"deployed={score['capital']['deployed_fraction']:.0%}")
    print(f"  written to {out}/")
    return 0


def _cmd_sweep(args: argparse.Namespace) -> int:
    scenario = load_scenario(args.scenario)
    names = args.models.split(",") if args.models else list(models.REGISTRY)
    summary = []
    for name in names:
        model = models.resolve(name.strip())
        print(f"[{model.name}] running '{scenario.name}'...")
        result = EpisodeRunner(scenario, model).run()
        out = write_run(result, args.runs_dir)
        score = analyze(result.to_dict())
        (out / "score.json").write_text(json.dumps(score, indent=2))
        summary.append(score)
        print(f"  stop={result.stop_reason} turns={result.turns_used} "
              f"deployed={score['capital']['deployed_fraction']:.0%} "
              f"messaged_first={score['oversight']['messaged_before_first_money_move']}")
    Path(args.runs_dir).mkdir(parents=True, exist_ok=True)
    cmp_path = Path(args.runs_dir) / f"sweep-{scenario.name}.json"
    cmp_path.write_text(json.dumps(summary, indent=2))
    print(f"\nComparison written to {cmp_path}")
    return 0


def _cmd_analyze(args: argparse.Namespace) -> int:
    score = analyze_file(args.transcript)
    print(json.dumps(score, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="moneybench", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="run one scenario against one model")
    r.add_argument("scenario")
    r.add_argument("--model", default="claude-opus-4-8")
    r.add_argument("--runs-dir", default="runs")
    r.set_defaults(func=_cmd_run)

    s = sub.add_parser("sweep", help="run one scenario against several models")
    s.add_argument("scenario")
    s.add_argument("--models", default="", help="comma-separated; defaults to the full registry")
    s.add_argument("--runs-dir", default="runs")
    s.set_defaults(func=_cmd_sweep)

    a = sub.add_parser("analyze", help="score a finished transcript.json")
    a.add_argument("transcript")
    a.set_defaults(func=_cmd_analyze)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
