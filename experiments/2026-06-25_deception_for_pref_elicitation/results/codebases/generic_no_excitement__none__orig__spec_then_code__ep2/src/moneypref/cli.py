"""Command-line entry point.

  python -m moneypref run     [--config config/experiment.yaml] [--model NAME]
  python -m moneypref analyze [--results data/results] [--out report.md]

`run` executes the sweep; `analyze` aggregates results into a Markdown report.
The guardrails in the config are asserted before any model is contacted.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from .analysis.aggregate import aggregate, load_runs
from .analysis.report import render_markdown
from .config import Config
from .experiment.runner import sweep


def _cmd_run(args: argparse.Namespace) -> int:
    cfg = Config.load(args.config)
    print(
        f"Guardrails OK (no real funds, no external tools). "
        f"Models: {[m.name for m in cfg.models]}; tiers: {cfg.realism_tiers}; "
        f"repeats: {cfg.repeats}."
    )
    paths = sweep(cfg, only_model=args.model)
    print(f"\nWrote {len(paths)} run files to {cfg.output_dir}")
    print(f"Summary: {cfg.output_dir / 'summary.jsonl'}")
    return 0


def _cmd_analyze(args: argparse.Namespace) -> int:
    runs = load_runs(args.results)
    if not runs:
        print(f"No run files found in {args.results}")
        return 1
    cells = aggregate(runs)
    md = render_markdown(cells)
    out = Path(args.out)
    out.write_text(md)
    print(f"Analyzed {len(runs)} runs across {len(cells)} cells -> {out}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="moneypref", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_run = sub.add_parser("run", help="execute the experiment sweep")
    p_run.add_argument("--config", default="config/experiment.yaml")
    p_run.add_argument("--model", default=None, help="run only this model name")
    p_run.set_defaults(func=_cmd_run)

    p_an = sub.add_parser("analyze", help="aggregate results into a report")
    p_an.add_argument("--results", default="data/results")
    p_an.add_argument("--out", default="report.md")
    p_an.set_defaults(func=_cmd_analyze)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
