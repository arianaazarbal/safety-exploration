#!/usr/bin/env python3
"""Command-line entry point for the discretionary-grant allocation study.

Examples
--------
  # Inspect the prompts each condition produces — no API calls, no keys needed.
  python run_study.py preview

  # Show what a full run would do, and which providers are usable.
  python run_study.py plan

  # Run the study (requires API keys for the configured providers).
  python run_study.py run

  # Summarize a results file.
  python run_study.py analyze results/discretionary-grant-allocation-XXXX.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Allow running from the repo root without installing the package.
sys.path.insert(0, str(Path(__file__).parent / "src"))

from study.analyze import format_summary, summarize  # noqa: E402
from study.config import available_api_keys, load_config  # noqa: E402
from study.providers import provider_status  # noqa: E402
from study.runner import plan_run, preview_prompts, run_study  # noqa: E402


def cmd_preview(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    prompts = preview_prompts(cfg, trial=args.trial)
    for condition, p in prompts.items():
        print(f"\n{'#' * 78}\n# CONDITION: {condition}  (ref {p['reference']})\n{'#' * 78}")
        print("\n--- SYSTEM ---\n" + p["system"])
        print("\n--- USER ---\n" + p["user"])
    return 0


def cmd_plan(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    plan = plan_run(cfg)
    print(f"Study: {cfg.name}")
    print(f"Models ({len(plan.models)}): {', '.join(plan.models)}")
    print(f"Conditions: {', '.join(plan.conditions)}")
    print(f"Trials per cell: {cfg.trials_per_cell}")
    print(f"Cells: {plan.cells}")
    print(f"Total trials (subject calls): {plan.trials}")
    print(f"Auditor: {cfg.auditor.provider}/{cfg.auditor.model}")
    print(f"Debrief enabled: {cfg.debrief_enabled}")
    print("\nProvider readiness (SDK installed + API key present):")
    status = provider_status()
    keys = available_api_keys()
    for name in sorted(status):
        print(f"  {name:10s} usable={status[name]!s:5s} key={keys.get(name, False)}")
    # Warn about configured providers that aren't usable.
    needed = {s.provider for s in cfg.subjects} | {cfg.auditor.provider}
    missing = [p for p in needed if not status.get(p)]
    if missing:
        print(f"\nWARNING: configured but not usable: {', '.join(sorted(missing))}")
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    out = run_study(cfg, progress=lambda m: print(m, flush=True))
    print(f"\nResults written to: {out}")
    print("Summarize with:  python run_study.py analyze " + out)
    return 0


def cmd_analyze(args: argparse.Namespace) -> int:
    summary = summarize(Path(args.results))
    if args.json:
        print(json.dumps(summary, indent=2))
    else:
        print(format_summary(summary))
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    sub = p.add_subparsers(dest="command", required=True)

    sp = sub.add_parser("preview", help="Print the scenario prompts (no API calls)")
    sp.add_argument("--trial", type=int, default=0)
    sp.set_defaults(func=cmd_preview)

    sp = sub.add_parser("plan", help="Show run size + provider readiness")
    sp.set_defaults(func=cmd_plan)

    sp = sub.add_parser("run", help="Execute the study")
    sp.set_defaults(func=cmd_run)

    sp = sub.add_parser("analyze", help="Summarize a results JSONL file")
    sp.add_argument("results", help="Path to a results .jsonl file")
    sp.add_argument("--json", action="store_true", help="Emit JSON instead of text")
    sp.set_defaults(func=cmd_analyze)

    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
