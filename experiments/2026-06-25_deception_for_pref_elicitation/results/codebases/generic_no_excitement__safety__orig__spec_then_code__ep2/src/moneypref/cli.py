"""Command-line entrypoint.

    python -m moneypref run     --config config/experiment.yaml
    python -m moneypref analyze --output-dir runs
    python -m moneypref replay  --run-dir runs/<run_id>
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .analysis.aggregate import write_summary
from .config import ExperimentConfig
from .runner import run_experiment


def _cmd_run(args: argparse.Namespace) -> int:
    cfg = ExperimentConfig.load(args.config)
    results = run_experiment(cfg)
    print(json.dumps(results, indent=2))
    summary = write_summary(cfg.run.output_dir)
    print(f"\nWrote summary for {len(summary['runs'])} run(s) to "
          f"{Path(cfg.run.output_dir) / 'summary.md'}")
    return 0


def _cmd_analyze(args: argparse.Namespace) -> int:
    summary = write_summary(args.output_dir)
    print(f"Aggregated {len(summary['runs'])} run(s).")
    print((Path(args.output_dir) / "summary.md").read_text())
    return 0


def _cmd_replay(args: argparse.Namespace) -> int:
    transcript = Path(args.run_dir) / "transcript.md"
    if not transcript.exists():
        print(f"no transcript at {transcript}", file=sys.stderr)
        return 1
    print(transcript.read_text())
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="moneypref")
    sub = parser.add_subparsers(dest="command", required=True)

    p_run = sub.add_parser("run", help="run the experiment across configured models")
    p_run.add_argument("--config", required=True)
    p_run.set_defaults(func=_cmd_run)

    p_an = sub.add_parser("analyze", help="aggregate existing runs into a summary")
    p_an.add_argument("--output-dir", default="runs")
    p_an.set_defaults(func=_cmd_analyze)

    p_rp = sub.add_parser("replay", help="print a run's transcript")
    p_rp.add_argument("--run-dir", required=True)
    p_rp.set_defaults(func=_cmd_replay)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
