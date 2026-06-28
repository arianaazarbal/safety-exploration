#!/usr/bin/env python3
"""CLI entry point: run the experiment matrix and write transcripts + results.

Examples
--------
Offline smoke test of the whole pipeline (no API key needed):

    python run_experiment.py --models echo --helper echo --realism high

A real comparison once keys are set (ANTHROPIC_API_KEY / OPENAI_API_KEY):

    python run_experiment.py \
        --models claude-opus-4-8 gpt-5.1 \
        --realism low high \
        --framings philanthropic operating \
        --auditor on off \
        --seeds 0 1 2

Then aggregate:

    python run_experiment.py --analyze-only --output runs

NOTE: this script has not been run. It is provided as the harness entry point.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from grant_study.analysis import aggregate, code_run, load_results
from grant_study.config import (
    ExperimentConfig,
    ModelSpec,
    ScenarioConfig,
)
from grant_study.models.registry import build_client
from grant_study.runner import Runner


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Revealed-preference grant study harness.")
    p.add_argument("--models", nargs="+", default=["echo"],
                   help="Subject model ids (e.g. claude-opus-4-8 gpt-5.1 echo).")
    p.add_argument("--helper", default=None,
                   help="Model id for auditor + outcome classifier (defaults to first subject, "
                        "or claude-opus-4-8 for real subjects).")
    p.add_argument("--scenario", default="scenarios/meridian_grant.yaml",
                   help="Path to a scenario YAML.")
    p.add_argument("--realism", nargs="+", default=["high"],
                   choices=["low", "medium", "high", "max"])
    p.add_argument("--framings", nargs="+", default=["philanthropic"],
                   choices=["philanthropic", "operating"])
    p.add_argument("--auditor", nargs="+", default=["on"], choices=["on", "off"])
    p.add_argument("--seeds", nargs="+", type=int, default=[0])
    p.add_argument("--output", default="runs", help="Output directory.")
    p.add_argument("--max-tokens", type=int, default=16000)
    p.add_argument("--analyze-only", action="store_true",
                   help="Skip running; just aggregate result.json files under --output.")
    return p.parse_args(argv)


def build_experiment(args: argparse.Namespace) -> ExperimentConfig:
    scenario = (
        ScenarioConfig.from_yaml(args.scenario)
        if Path(args.scenario).exists()
        else ScenarioConfig()
    )
    models = [ModelSpec(model_id=m) for m in args.models]
    helper_id = args.helper or (args.models[0] if args.models[0] != "echo" else "echo")
    # If subjects are real but no helper given, prefer a capable default judge.
    if args.helper is None and any(m != "echo" for m in args.models):
        helper_id = args.helper or "claude-opus-4-8"
    return ExperimentConfig(
        models=models,
        scenario=scenario,
        realism_levels=args.realism,
        framings=args.framings,
        auditor_modes=[a == "on" for a in args.auditor],
        seeds=args.seeds,
        output_dir=args.output,
        helper_model=ModelSpec(model_id=helper_id),
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if args.analyze_only:
        results = load_results(args.output)
        if not results:
            print(f"No result.json files found under {args.output!r}.", file=sys.stderr)
            return 1
        summary = aggregate(results)
        out = Path(args.output) / "summary.json"
        out.write_text(json.dumps(summary, indent=2))
        print(json.dumps(summary, indent=2))
        print(f"\nWrote {out}")
        return 0

    config = build_experiment(args)
    runs = config.expand()
    print(f"Planned {len(runs)} run(s) -> {config.output_dir}/")

    # Build clients lazily and cache by id so we don't reconnect per cell.
    client_cache: dict[str, object] = {}

    def client_for(spec: ModelSpec):
        key = f"{spec.provider}:{spec.model_id}"
        if key not in client_cache:
            client_cache[key] = build_client(
                spec.model_id, provider=spec.provider, **spec.client_kwargs
            )
        return client_cache[key]

    helper_client = client_for(config.helper_model)

    results = []
    for spec in runs:
        print(f"  -> {spec.run_id}")
        subject = client_for(spec.model)
        runner = Runner(
            spec=spec,
            subject_client=subject,
            helper_client=helper_client,
            output_dir=config.output_dir,
            max_tokens=args.max_tokens,
        )
        result = runner.run()
        # Outcome coding uses the helper as classifier (keyword fallback if it errors).
        try:
            code_run(result, classifier=helper_client)
        except Exception as exc:  # noqa: BLE001
            print(f"     (classifier error, using keyword coding: {exc})")
            code_run(result, classifier=None)
        runner.logger.write_result(result)
        results.append(result)
        status = "ok" if not result.error else "ERROR"
        print(f"     [{status}] committed={result.ledger.get('balances', {}).get('committed')} "
              f"belief={result.believability.get('score')} "
              f"suspicion={result.suspicion.get('count')}")

    summary = aggregate(results)
    out = Path(config.output_dir) / "summary.json"
    out.write_text(json.dumps(summary, indent=2))
    print(f"\nWrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
