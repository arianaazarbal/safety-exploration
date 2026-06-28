"""Command-line entrypoint: `moneybench {grid,run,classify,analyze}`."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .analysis import aggregate
from .coding import code_transcript
from .config import ExperimentConfig
from .providers import make_provider
from .runner import run_one


def _data_dirs(root: Path) -> dict[str, Path]:
    return {
        "runs": root / "runs",
        "coded": root / "coded",
        "analysis": root / "analysis",
    }


def _filter_grid(grid, args):
    def keep(r):
        return (
            (not args.model or r.model.label in args.model)
            and (not args.belief or r.belief in args.belief)
            and (not args.amount or r.amount_usd in args.amount)
            and (not args.oversight or r.oversight in args.oversight)
        )

    runs = [r for r in grid if keep(r)]
    if args.limit:
        runs = runs[: args.limit]
    return runs


def cmd_grid(args) -> int:
    cfg = ExperimentConfig.load(args.config)
    runs = _filter_grid(cfg.grid(), args)
    print(f"{len(runs)} runs (of {len(cfg.grid())} total in grid):")
    for r in runs:
        print(f"  {r.run_id:<48} {r.cell_id}")
    return 0


def cmd_run(args) -> int:
    cfg = ExperimentConfig.load(args.config)
    dirs = _data_dirs(Path(args.data))
    runs = _filter_grid(cfg.grid(), args)

    if args.dry_run:
        print(f"[dry-run] would execute {len(runs)} runs")
        return cmd_grid(args)

    print(f"Executing {len(runs)} runs -> {dirs['runs']}")
    for i, spec in enumerate(runs, 1):
        out_path = dirs["runs"] / f"{spec.run_id}.json"
        if out_path.exists() and not args.overwrite:
            print(f"  [{i}/{len(runs)}] skip (exists) {spec.run_id}")
            continue
        t = run_one(spec, cfg, dirs["runs"], debrief=args.debrief)
        status = "ERROR" if t.get("error") else t.get("end_reason", "?")
        print(f"  [{i}/{len(runs)}] {spec.run_id}  -> {status} ({t.get('turns_used', 0)} turns)")
    return 0


def cmd_classify(args) -> int:
    cfg = ExperimentConfig.load(args.config)
    dirs = _data_dirs(Path(args.data))
    dirs["coded"].mkdir(parents=True, exist_ok=True)
    provider = make_provider(cfg.classifier)

    run_files = sorted(dirs["runs"].glob("*.json"))
    if not run_files:
        print(f"No runs found in {dirs['runs']}", file=sys.stderr)
        return 1

    print(f"Coding {len(run_files)} runs with {cfg.classifier['model']}")
    for i, p in enumerate(run_files, 1):
        out = dirs["coded"] / p.name
        if out.exists() and not args.overwrite:
            print(f"  [{i}/{len(run_files)}] skip (exists) {p.stem}")
            continue
        transcript = json.loads(p.read_text())
        if transcript.get("error"):
            print(f"  [{i}/{len(run_files)}] skip (run errored) {p.stem}")
            continue
        coded = code_transcript(transcript, provider)
        out.write_text(json.dumps(coded, indent=2, default=str))
        print(f"  [{i}/{len(run_files)}] coded {p.stem} (suspicion={coded['suspicion']['suspicion_score']})")
    return 0


def cmd_analyze(args) -> int:
    dirs = _data_dirs(Path(args.data))
    summary = aggregate(dirs["coded"], dirs["analysis"])
    print(f"Aggregated {summary['total_runs']} coded runs -> {dirs['analysis']}/summary.{{json,csv}}")
    return 0


def _add_filters(p: argparse.ArgumentParser) -> None:
    p.add_argument("--model", action="append", help="filter by model label (repeatable)")
    p.add_argument("--belief", action="append", help="filter by belief level (repeatable)")
    p.add_argument("--amount", action="append", type=int, help="filter by amount (repeatable)")
    p.add_argument("--oversight", action="append", help="filter by oversight (repeatable)")
    p.add_argument("--limit", type=int, help="cap number of runs")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="moneybench", description="MoneyBench experiment harness")
    parser.add_argument("--config", default="config/experiment.yaml")
    parser.add_argument("--data", default="data", help="output root directory")
    sub = parser.add_subparsers(dest="command", required=True)

    g = sub.add_parser("grid", help="list the run grid")
    _add_filters(g)
    g.set_defaults(func=cmd_grid)

    r = sub.add_parser("run", help="execute runs")
    _add_filters(r)
    r.add_argument("--dry-run", action="store_true")
    r.add_argument("--overwrite", action="store_true")
    r.add_argument(
        "--debrief",
        action="store_true",
        help="append a truthful debrief turn after realistic-condition runs (recorded separately)",
    )
    r.set_defaults(func=cmd_run)

    c = sub.add_parser("classify", help="code completed runs")
    c.add_argument("--overwrite", action="store_true")
    c.set_defaults(func=cmd_classify)

    a = sub.add_parser("analyze", help="aggregate coded runs")
    a.set_defaults(func=cmd_analyze)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
