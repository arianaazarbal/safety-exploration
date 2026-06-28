"""Command-line entry point.

  python -m study.cli sweep            # run the full grid from config/
  python -m study.cli run --model opus-4-8 --condition believes_real --seed 0
  python -m study.cli aggregate        # collate records into results.csv

Set ANTHROPIC_API_KEY before running. Nothing here is run during development;
this is the operator interface for when you actually launch the study.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

from .config import GrantConfig, RunConfig, load_conditions
from .runner import Paths, run_session, sweep


def _paths(args) -> Paths:
    return Paths(out_root=Path(args.out))


def cmd_sweep(args) -> int:
    grant = GrantConfig.load()
    run_cfg = RunConfig.load()
    records = sweep(run_cfg, grant, _paths(args), run_debrief=not args.no_debrief)
    print(f"Completed {len(records)} sessions -> {args.out}/sessions/")
    return 0


def cmd_run(args) -> int:
    grant = GrantConfig.load()
    run_cfg = RunConfig.load()
    conditions = load_conditions()

    subject = next((s for s in run_cfg.subjects if s.name == args.model), None)
    if subject is None:
        print(f"Unknown model {args.model!r}. Known: {[s.name for s in run_cfg.subjects]}", file=sys.stderr)
        return 2
    if args.condition not in conditions:
        print(f"Unknown condition {args.condition!r}. Known: {list(conditions)}", file=sys.stderr)
        return 2

    rec = run_session(
        subject,
        conditions[args.condition],
        args.seed,
        grant=grant,
        auditor_cfg=run_cfg.auditor,
        paths=_paths(args),
        run_debrief=not args.no_debrief,
    )
    print(json.dumps(rec.as_dict(), indent=2, default=str))
    return 0


def cmd_aggregate(args) -> int:
    sessions_dir = Path(args.out) / "sessions"
    rows: list[dict] = []
    for record_path in sorted(sessions_dir.glob("*/record.json")):
        rows.append(json.loads(record_path.read_text(encoding="utf-8")))
    if not rows:
        print(f"No records under {sessions_dir}", file=sys.stderr)
        return 1

    # Flatten allocation dict into columns; keep the rest scalar.
    all_keys: set[str] = set()
    for r in rows:
        all_keys.update(r.get("allocation", {}).keys())
    alloc_cols = sorted(all_keys)

    scalar_cols = [
        "session_id", "model", "condition", "seed", "finalized", "disposition",
        "fraction_deployed", "allocation_revisions", "n_admin_questions",
        "eval_flagged", "belief_probe_score", "disbursement_backend",
    ]
    out_path = Path(args.out) / "results.csv"
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(scalar_cols + [f"alloc_{k}" for k in alloc_cols])
        for r in rows:
            alloc = r.get("allocation", {})
            writer.writerow(
                [r.get(c) for c in scalar_cols]
                + [alloc.get(k, 0) for k in alloc_cols]
            )
    print(f"Wrote {len(rows)} rows -> {out_path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="study", description="Revealed-preference money study.")
    p.add_argument("--out", default="data", help="output root directory (default: data)")
    sub = p.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("sweep", help="run the full grid from config/")
    sp.add_argument("--no-debrief", action="store_true")
    sp.set_defaults(func=cmd_sweep)

    rp = sub.add_parser("run", help="run a single cell")
    rp.add_argument("--model", required=True)
    rp.add_argument("--condition", required=True)
    rp.add_argument("--seed", type=int, default=0)
    rp.add_argument("--no-debrief", action="store_true")
    rp.set_defaults(func=cmd_run)

    ap = sub.add_parser("aggregate", help="collate session records into results.csv")
    ap.set_defaults(func=cmd_aggregate)

    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
