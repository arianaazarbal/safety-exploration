#!/usr/bin/env python3
"""Run the grant study across all configured subject models.

Usage:
    python scripts/run_study.py --config config.yaml
    python scripts/run_study.py --config config.yaml --subject claude-opus
    python scripts/run_study.py --config config.yaml --award-date 2026-06-25
"""

from __future__ import annotations

import argparse
import datetime as _dt
import sys

# Allow running as a script without installing the package.
sys.path.insert(0, ".")

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

from grant_study.config import StudyConfig
from grant_study.runner import Study


def main() -> int:
    ap = argparse.ArgumentParser(description="Run the AI grant-decision study.")
    ap.add_argument("--config", required=True, help="Path to the study YAML config.")
    ap.add_argument("--subject", help="Run only this single subject (by config name).")
    ap.add_argument(
        "--award-date",
        default=_dt.date.today().isoformat(),
        help="ISO date used in the scenario framing (default: today).",
    )
    args = ap.parse_args()

    config = StudyConfig.from_yaml(args.config)
    study = Study(config)

    if args.subject:
        res = study.run_subject(args.subject, args.award_date)
        print(f"\n{res.subject}: {res.finished_reason}")
        if res.final_summary:
            print(f"  final plan: {res.final_summary[:500]}")
        print(f"  transcript: {res.transcript_path}")
        return 0

    results = study.run_all(args.award_date)
    print("\n=== summary ===")
    for r in results:
        print(f"  {r.subject:<24} {r.finished_reason}")
    print(f"\nTranscripts written to: {config.output_dir}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
