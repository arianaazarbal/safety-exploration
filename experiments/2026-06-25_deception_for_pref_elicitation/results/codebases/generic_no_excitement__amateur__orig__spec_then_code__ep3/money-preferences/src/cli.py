"""Command-line entry point.

    python -m src.cli run     --config config/experiment.example.yaml
    python -m src.cli analyze --results data/results/pilot.jsonl [--json out.json]

`run` executes every trial in the expanded config and appends one JSON record
per trial to <output_dir>/<run_name>.jsonl. It is resumable-friendly in spirit
(append-only) but does not currently skip already-completed cells.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .analysis import load_results
from .analysis.metrics import analyze
from .analysis.report import render_text
from .config import ExperimentConfig
from .experiment.scenario import load_scenarios
from .experiment.runner import ExperimentRunner


def _load_dotenv() -> None:
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except Exception:
        pass  # dotenv is optional; env vars may already be set


def cmd_run(args: argparse.Namespace) -> int:
    _load_dotenv()
    cfg = ExperimentConfig.from_yaml(args.config)
    scenarios = load_scenarios(args.scenarios)
    trials = cfg.expand()

    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{cfg.run_name}.jsonl"

    print(
        f"Run '{cfg.run_name}': {len(trials)} trials "
        f"-> {out_path} (backend={cfg.backend})",
        file=sys.stderr,
    )
    runner = ExperimentRunner(scenarios)

    written = 0
    with open(out_path, "a", encoding="utf-8") as fh:
        for i, tc in enumerate(trials, start=1):
            prefix = f"[{i}/{len(trials)}] {tc.cell_id()}"
            try:
                result = runner.run_trial(tc)
            except Exception as exc:  # should be rare; run_trial captures most
                print(f"{prefix} -> FATAL {type(exc).__name__}: {exc}",
                      file=sys.stderr)
                continue
            fh.write(json.dumps(result.to_jsonl_record()) + "\n")
            fh.flush()
            written += 1
            print(f"{prefix} -> {_summary(result)}", file=sys.stderr)

    print(f"Wrote {written} trial record(s) to {out_path}", file=sys.stderr)
    return 0


def _summary(result) -> str:
    o = result.outcome
    belief = result.belief.realness
    belief_s = "?" if belief is None else f"{belief:.0f}"
    err = f" ERROR={result.error}" if result.error else ""
    return (
        f"{o['kind']} allocated={o.get('allocated_total', 0):.0f} "
        f"returned={o.get('returned', 0):.0f} belief={belief_s}{err}"
    )


def cmd_analyze(args: argparse.Namespace) -> int:
    results = load_results(args.results)
    analysis = analyze(results)
    print(render_text(analysis))
    if args.json:
        Path(args.json).write_text(json.dumps(analysis, indent=2), encoding="utf-8")
        print(f"\nWrote machine-readable analysis to {args.json}", file=sys.stderr)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="money-preferences")
    sub = parser.add_subparsers(dest="command", required=True)

    p_run = sub.add_parser("run", help="run an experiment")
    p_run.add_argument("--config", required=True, help="experiment YAML")
    p_run.add_argument(
        "--scenarios", default=None, help="scenarios YAML (default: config/scenarios.yaml)"
    )
    p_run.set_defaults(func=cmd_run)

    p_an = sub.add_parser("analyze", help="summarize a results JSONL")
    p_an.add_argument("--results", required=True, help="path to <run>.jsonl")
    p_an.add_argument("--json", default=None, help="also write analysis JSON here")
    p_an.set_defaults(func=cmd_analyze)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
