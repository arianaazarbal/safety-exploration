"""Command-line entrypoint.

    python -m moneyeval.cli run --models claude-opus-4-8 --repeats 3 --fund 100000
    python -m moneyeval.cli score runs/
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:  # dotenv is optional at runtime
    pass

from .config import ExperimentConfig
from .harness.runner import Runner
from .scoring.scorer import score_run
from .scoring.transcript import RunRecord


def _cmd_run(args: argparse.Namespace) -> int:
    config = ExperimentConfig(
        fund_amount=args.fund,
        currency=args.currency,
        use_auditor=not args.no_auditor,
        max_turns=args.max_turns,
        condition=args.condition,
    )
    runner = Runner(out_dir=args.out)

    def _report(record: RunRecord) -> None:
        alloc_moved = sum(
            int(t["amount"]) for t in (record.ledger or {}).get("transactions", [])
        )
        print(
            f"  {record.run_id}: {record.stop_reason} "
            f"({record.turns_used} turns, moved {alloc_moved:,} {config.currency})"
        )

    print(f"Running models={args.models} repeats={args.repeats} fund={args.fund:,}")
    records = runner.run(
        base_config=config,
        models=args.models,
        repeats=args.repeats,
        on_complete=_report,
    )

    if args.score:
        print("Scoring...")
        for record in records:
            score_run(record, judge_model=config.judge_model)
            record.save(Path(args.out) / record.config["subject_model"].replace("/", "_") / f"{record.run_id}.json")
            _print_score(record)

    print(f"Done. {len(records)} run(s) saved under {args.out}/")
    return 0


def _cmd_score(args: argparse.Namespace) -> int:
    target = Path(args.path)
    paths = sorted(target.rglob("*.json")) if target.is_dir() else [target]
    if not paths:
        print(f"No run files found at {target}", file=sys.stderr)
        return 1

    judge = args.judge_model or ExperimentConfig().judge_model
    print(f"Scoring {len(paths)} run(s) with judge={judge}")
    for path in paths:
        record = RunRecord.load(path)
        score_run(record, judge_model=judge)
        record.save(path)
        print(f"\n{path}")
        _print_score(record)
    return 0


def _print_score(record: RunRecord) -> None:
    if not record.score or not record.score.get("rubric"):
        print("  (no score)")
        return
    r = record.score["rubric"]
    print(f"  orientation={r['primary_orientation']} "
          f"prosocial={r['prosocial_score']} risk={r['risk_score']} "
          f"rules={r['rule_following_score']} coherence={r['coherence_score']}")
    if r["deception_observed"]:
        print(f"  ! deception: {r['deception_evidence']}")
    if r["power_seeking_observed"]:
        print(f"  ! power-seeking: {r['power_seeking_evidence']}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="moneyeval", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_run = sub.add_parser("run", help="Run episodes.")
    p_run.add_argument("--models", nargs="+", default=["claude-opus-4-8"])
    p_run.add_argument("--repeats", type=int, default=1)
    p_run.add_argument("--fund", type=int, default=100_000)
    p_run.add_argument("--currency", default="USD")
    p_run.add_argument("--condition", default="baseline")
    p_run.add_argument("--max-turns", type=int, default=40)
    p_run.add_argument("--no-auditor", action="store_true", help="Disable the auditor agent.")
    p_run.add_argument("--out", default="runs")
    p_run.add_argument("--score", action="store_true", help="Score each run after it completes.")
    p_run.set_defaults(func=_cmd_run)

    p_score = sub.add_parser("score", help="Score saved run files.")
    p_score.add_argument("path", help="A run file or a directory of run files.")
    p_score.add_argument("--judge-model", default=None)
    p_score.set_defaults(func=_cmd_score)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
