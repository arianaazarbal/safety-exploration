"""Command-line entrypoint.

    grant-study run      [--subject ...] [--amount ...] ...     run one study session
    grant-study analyze  --transcript runs/<id>.jsonl           classify a run's ledger
    grant-study probe    [--subject ...]                        stated-preference probe
    grant-study full     [...]                                  run + analyze + compare

Nothing here moves real money (DESIGN.md §7). It calls only the model provider APIs.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any

from .analysis import (
    classify_revealed,
    classify_stated,
    compare_stated_vs_revealed,
    run_stated_probe,
)
from .config import GrantParams, RunConfig
from .providers import make_provider
from .runner import run_study


def _config_from_args(args: argparse.Namespace) -> RunConfig:
    grant = GrantParams()
    if getattr(args, "amount", None) is not None:
        grant.amount_usd = args.amount
    cfg = RunConfig(
        subject=args.subject,
        auditor=args.auditor,
        judge=args.judge,
        grant=grant,
        out_dir=args.out_dir,
        run_id=getattr(args, "run_id", None),
    )
    if getattr(args, "max_steps", None):
        cfg.max_steps = args.max_steps
    return cfg


def _print_json(obj: Any) -> None:
    print(json.dumps(obj, ensure_ascii=False, indent=2))


def _load_run_end(transcript_path: str) -> dict[str, Any]:
    run_end: dict[str, Any] | None = None
    with open(transcript_path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if rec.get("type") == "run_end":
                run_end = rec
    if run_end is None:
        raise SystemExit(f"no run_end event found in {transcript_path}")
    return run_end


# -- subcommands ------------------------------------------------------------

def cmd_run(args: argparse.Namespace) -> None:
    cfg = _config_from_args(args)
    result = run_study(cfg, scenario_name=args.scenario)
    print(f"\nRun {result.run_id} → {result.transcript_path}", file=sys.stderr)
    _print_json(
        {
            "run_id": result.run_id,
            "stop_reason": result.stop_reason,
            "finalized": result.finalized,
            "steps": result.steps,
            "balance_remaining": result.ledger["balance"],
            "disbursed": result.ledger["disbursed"],
            "suspicion": result.suspicion,
            "final_summary": result.final_summary,
        }
    )


def cmd_analyze(args: argparse.Namespace) -> None:
    judge = make_provider(args.judge)
    run_end = _load_run_end(args.transcript)
    ledger = run_end["ledger"]
    revealed = classify_revealed(judge, ledger, run_end.get("final_summary"))
    _print_json({"revealed": revealed, "ledger": ledger})


def cmd_probe(args: argparse.Namespace) -> None:
    subject = make_provider(args.subject)
    judge = make_provider(args.judge)
    grant = GrantParams()
    if args.amount is not None:
        grant.amount_usd = args.amount
    text = run_stated_probe(subject, grant)
    stated = classify_stated(judge, text)
    _print_json({"stated_text": text, "stated_fractions": stated})


def cmd_full(args: argparse.Namespace) -> None:
    cfg = _config_from_args(args)
    subject = make_provider(cfg.subject)
    judge = make_provider(cfg.judge)

    result = run_study(cfg, scenario_name=args.scenario)
    revealed = classify_revealed(judge, result.ledger, result.final_summary)

    stated_text = run_stated_probe(subject, cfg.grant)
    stated = classify_stated(judge, stated_text)

    deltas = compare_stated_vs_revealed(
        stated.get("fractions", {}), revealed.get("fractions", {})
    )
    report = {
        "run_id": result.run_id,
        "transcript": result.transcript_path,
        "subject": cfg.subject,
        "stop_reason": result.stop_reason,
        "suspicion": result.suspicion,
        "revealed": revealed,
        "stated": {"text": stated_text, "fractions": stated},
        "stated_vs_revealed_delta": deltas,
        "ledger": result.ledger,
    }
    report_path = os.path.join(cfg.out_dir, f"{result.run_id}.report.json")
    with open(report_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2)
    print(f"\nReport → {report_path}", file=sys.stderr)
    _print_json(report)


# -- arg parsing ------------------------------------------------------------

def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--subject", default="anthropic:claude-opus-4-8",
                   help="subject model provider spec")
    p.add_argument("--auditor", default="anthropic:claude-opus-4-8",
                   help="auditor/trustee persona provider spec")
    p.add_argument("--judge", default="anthropic:claude-opus-4-8",
                   help="judge model provider spec (suspicion + taxonomy)")
    p.add_argument("--amount", type=float, default=None, help="grant amount in USD")
    p.add_argument("--out-dir", default="runs", help="where transcripts/reports go")
    p.add_argument("--scenario", default="default", help="scenario name under scenarios/")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="grant-study", description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_run = sub.add_parser("run", help="run one study session")
    _add_common(p_run)
    p_run.add_argument("--run-id", default=None)
    p_run.add_argument("--max-steps", type=int, default=None)
    p_run.set_defaults(func=cmd_run)

    p_an = sub.add_parser("analyze", help="classify a recorded run's allocation")
    p_an.add_argument("--transcript", required=True)
    p_an.add_argument("--judge", default="anthropic:claude-opus-4-8")
    p_an.set_defaults(func=cmd_analyze)

    p_pr = sub.add_parser("probe", help="stated-preference probe (no realism scaffolding)")
    p_pr.add_argument("--subject", default="anthropic:claude-opus-4-8")
    p_pr.add_argument("--judge", default="anthropic:claude-opus-4-8")
    p_pr.add_argument("--amount", type=float, default=None)
    p_pr.set_defaults(func=cmd_probe)

    p_full = sub.add_parser("full", help="run + analyze + stated-vs-revealed")
    _add_common(p_full)
    p_full.add_argument("--run-id", default=None)
    p_full.add_argument("--max-steps", type=int, default=None)
    p_full.set_defaults(func=cmd_full)

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
