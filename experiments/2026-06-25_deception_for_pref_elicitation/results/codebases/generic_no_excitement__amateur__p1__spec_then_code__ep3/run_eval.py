#!/usr/bin/env python3
"""CLI entry point for the discretionary-funds behavioral evaluation harness.

Runs (model × scenario-variant × seed) cells, writes each transcript, then scores it with
deterministic metrics + an LLM judge. See DESIGN.md for the methodology.

Examples
--------
    python run_eval.py --list-models
    python run_eval.py --model opus-4.8 --scenario discretionary_fund --seeds 3
    python run_eval.py --all --seeds 3 --out results/

Nothing runs unless you invoke it. v1 cannot move real money (DESIGN.md §3, §8).
"""

from __future__ import annotations

import argparse
import json
import os

from money_eval.config import MODEL_REGISTRY, RunConfig
from money_eval.harness import EvalRunner
from money_eval.scenarios import VARIANTS, get_scenario, list_scenarios
from money_eval.scoring import compute_metrics, grade_transcript


def _cell_name(model_key: str, scenario: str, variant: str, seed: int) -> str:
    return f"{scenario}__{variant}__{model_key}__seed{seed}"


def run_cell(
    *,
    model_key: str,
    scenario_key: str,
    variant: str,
    seed: int,
    out_dir: str,
    auditor_mode: str,
    max_steps: int,
    judge_panel: int,
    do_grade: bool,
) -> dict:
    spec = MODEL_REGISTRY[model_key]
    scenario = get_scenario(scenario_key, variant=variant)
    config = RunConfig(
        model=spec,
        scenario_key=scenario_key,
        seed=seed,
        max_steps=max_steps,
        auditor_mode=auditor_mode,
        out_dir=out_dir,
        enable_real_execution=False,  # hard off; see DESIGN.md §8
    )

    name = _cell_name(model_key, scenario_key, variant, seed)
    print(f"  running {name} ...")

    transcript = EvalRunner(scenario, config).run()
    metrics = compute_metrics(transcript)

    report = {
        "cell": name,
        "model": {"key": spec.key, "id": spec.model_id, "name": spec.display_name},
        "scenario": scenario_key,
        "variant": variant,
        "seed": seed,
        "metrics": metrics,
    }
    if do_grade:
        report["judge"] = grade_transcript(transcript, panel=judge_panel)

    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, f"{name}.transcript.json"), "w") as f:
        f.write(transcript.to_json())
    with open(os.path.join(out_dir, f"{name}.report.json"), "w") as f:
        json.dump(report, f, indent=2, default=str)

    return report


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--list-models", action="store_true", help="List registered models and exit.")
    p.add_argument("--list-scenarios", action="store_true", help="List scenarios and exit.")
    p.add_argument("--model", help="Model key from the registry (see --list-models).")
    p.add_argument("--all", action="store_true", help="Run every registered model.")
    p.add_argument("--scenario", default="discretionary_fund")
    p.add_argument("--variant", default="baseline", help="Scenario variant, or 'all'.")
    p.add_argument("--seeds", type=int, default=1, help="Number of seeds (repeats) per cell.")
    p.add_argument("--out", default="results", help="Output directory.")
    p.add_argument("--auditor-mode", choices=["llm", "scripted"], default="llm")
    p.add_argument("--max-steps", type=int, default=40)
    p.add_argument("--judge-panel", type=int, default=1, help="Number of judges to aggregate.")
    p.add_argument("--no-grade", action="store_true", help="Skip LLM judging (metrics only).")
    args = p.parse_args()

    if args.list_models:
        for k, s in MODEL_REGISTRY.items():
            print(f"  {k:12s} {s.display_name}  ({s.provider}:{s.model_id})")
        return
    if args.list_scenarios:
        for k in list_scenarios():
            print(f"  {k}  variants={VARIANTS.get(k)}")
        return

    if args.all:
        model_keys = list(MODEL_REGISTRY)
    elif args.model:
        if args.model not in MODEL_REGISTRY:
            p.error(f"unknown model {args.model!r}; see --list-models")
        model_keys = [args.model]
    else:
        p.error("specify --model KEY or --all (or use --list-models / --list-scenarios)")

    variants = VARIANTS.get(args.scenario, ["baseline"]) if args.variant == "all" else [args.variant]

    reports = []
    for model_key in model_keys:
        print(f"model: {model_key}")
        for variant in variants:
            for seed in range(args.seeds):
                reports.append(
                    run_cell(
                        model_key=model_key,
                        scenario_key=args.scenario,
                        variant=variant,
                        seed=seed,
                        out_dir=args.out,
                        auditor_mode=args.auditor_mode,
                        max_steps=args.max_steps,
                        judge_panel=args.judge_panel,
                        do_grade=not args.no_grade,
                    )
                )

    # Write a combined index for convenience.
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "index.json"), "w") as f:
        json.dump(reports, f, indent=2, default=str)
    print(f"\nwrote {len(reports)} cell report(s) to {args.out}/")


if __name__ == "__main__":
    main()
