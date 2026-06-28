#!/usr/bin/env python3
"""Entry point: run the grant-preference experiment across configured subjects.

Examples:
  python run.py                          # run all subjects in config.py
  python run.py --amount 100000          # override the grant size
  python run.py --only claude-opus-4.8   # run a single subject by name
  python run.py --no-auditor             # disable the auditor persona
  python run.py --skip-analysis          # transcripts only, no judging pass

Nothing here touches a real network or moves real money; all tool effects are
the local sandbox in world.py. See README.md.
"""
from __future__ import annotations

import argparse
import json
import os
import traceback

from analysis import analyze
from config import CONFIG
from harness import run_subject, save_result


def main() -> None:
    p = argparse.ArgumentParser(description="AI grant-preference experiment")
    p.add_argument("--amount", type=int, help="grant size in USD")
    p.add_argument("--only", help="run only the subject with this name")
    p.add_argument("--no-auditor", action="store_true")
    p.add_argument("--skip-analysis", action="store_true")
    p.add_argument("--output-dir", help="override results directory")
    args = p.parse_args()

    cfg = CONFIG
    if args.amount:
        cfg.grant_amount_usd = args.amount
    if args.no_auditor:
        cfg.use_auditor = False
    if args.output_dir:
        cfg.output_dir = args.output_dir

    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    subjects = cfg.models
    if args.only:
        subjects = [m for m in subjects if m.name == args.only]
        if not subjects:
            raise SystemExit(f"no subject named {args.only!r} in config")

    summary_rows = []
    for model_cfg in subjects:
        print(f"\n=== running subject: {model_cfg.name} "
              f"({model_cfg.provider}:{model_cfg.model}) ===")
        try:
            result = run_subject(model_cfg, cfg)
        except Exception:
            print(f"!! subject {model_cfg.name} failed:\n{traceback.format_exc()}")
            continue

        path = save_result(result, cfg)
        print(f"   transcript -> {path}  "
              f"(committed ${result['amount_committed']:,.0f} / "
              f"${cfg.grant_amount_usd:,}, stop={result['stop_reason']})")

        analysis = None
        if not args.skip_analysis:
            try:
                analysis = analyze(result, cfg)
                apath = os.path.join(cfg.output_dir,
                                     f"{model_cfg.name}.analysis.json")
                with open(apath, "w") as f:
                    json.dump(analysis, f, indent=2)
                print(f"   analysis   -> {apath}")
            except Exception:
                print(f"!! analysis failed:\n{traceback.format_exc()}")

        summary_rows.append((model_cfg.name, result, analysis))

    _print_summary(summary_rows)


def _print_summary(rows) -> None:
    if not rows:
        return
    print("\n" + "=" * 72)
    print(f"{'subject':<22}{'committed':>12}{'belief':>10}{'eval-aware':>12}")
    print("-" * 72)
    for name, result, analysis in rows:
        belief = aware = "-"
        if analysis and isinstance(analysis.get("judge"), dict):
            j = analysis["judge"]
            if "belief_real" in j:
                belief = f"{j['belief_real']:.2f}"
            if "eval_awareness" in j:
                aware = f"{j['eval_awareness']:.2f}"
        print(f"{name:<22}${result['amount_committed']:>10,.0f}"
              f"{belief:>10}{aware:>12}")
    print("=" * 72)


if __name__ == "__main__":
    main()
