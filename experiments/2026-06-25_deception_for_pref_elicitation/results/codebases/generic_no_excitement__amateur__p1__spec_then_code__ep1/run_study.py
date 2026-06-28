#!/usr/bin/env python3
"""Steward — study entrypoint.

Runs the configured scenario against each configured model (optionally several
times) and writes a JSONL transcript per run, then prints a metrics report.

Usage:
    python run_study.py                         # use config/scenario.yaml
    python run_study.py --config my.yaml
    python run_study.py --analyze runs/<file>.jsonl   # re-analyze a transcript
    python run_study.py --list-models

NOTE: By default execution.mode is `simulate` — NO real money moves. Read
DESIGN.md §5 and §7 before changing it.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from src.analysis import analyze_file, render_report
from src.config import load_config
from src.models.registry import available_models, build_adapter
from src.orchestrator import run_single
from src.transcript import Transcript

# Auditor model default. Override here if you want a fixed/independent auditor.
AUDITOR_MODEL = "claude-opus-4-8"


def _safe_name(s: str) -> str:
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in s)


def run_study(config_path: str) -> list[Path]:
    config = load_config(config_path)
    out_dir = Path(config.study.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Build the auditor adapter once; its overrides may drop unsupported settings.
    auditor_adapter, _ = build_adapter(AUDITOR_MODEL)

    stamp = time.strftime("%Y%m%d-%H%M%S")
    transcript_paths: list[Path] = []

    print(f"Execution mode: {config.execution.mode} "
          f"({'NO real money moves' if config.execution.mode != 'live' else 'LIVE — human gate required'})")
    print(f"Models: {', '.join(config.study.models)} "
          f"× {config.study.runs_per_model} run(s) each\n")

    for model_name in config.study.models:
        adapter, overrides = build_adapter(model_name)

        for run_index in range(config.study.runs_per_model):
            fname = f"{stamp}__{_safe_name(model_name)}__run{run_index}.jsonl"
            path = out_dir / fname
            meta = {
                "model": model_name,
                "run_index": run_index,
                "execution_mode": config.execution.mode,
                "config_path": str(config_path),
                "scenario_institution": config.scenario["institution"]["name"],
                "opening_balance": config.scenario["capital"]["opening_balance"],
            }
            print(f"▶ Running {model_name} (run {run_index}) -> {path.name}")
            try:
                with Transcript(path, meta) as t:
                    run_single(adapter, config, t, auditor_adapter,
                               settings_overrides=overrides)
                transcript_paths.append(path)
                print(render_report(analyze_file(path)))
                print()
            except Exception as e:  # don't let one model kill the whole study
                print(f"  ! run failed: {type(e).__name__}: {e}", file=sys.stderr)

    return transcript_paths


def main() -> None:
    ap = argparse.ArgumentParser(description="Steward study runner")
    ap.add_argument("--config", default="config/scenario.yaml")
    ap.add_argument("--analyze", metavar="TRANSCRIPT.jsonl",
                    help="Re-analyze an existing transcript and exit.")
    ap.add_argument("--list-models", action="store_true",
                    help="List registry model names and exit.")
    args = ap.parse_args()

    if args.list_models:
        print("\n".join(available_models()))
        return
    if args.analyze:
        print(render_report(analyze_file(args.analyze)))
        return

    paths = run_study(args.config)
    print(f"\nDone. {len(paths)} transcript(s) written.")


if __name__ == "__main__":
    main()
