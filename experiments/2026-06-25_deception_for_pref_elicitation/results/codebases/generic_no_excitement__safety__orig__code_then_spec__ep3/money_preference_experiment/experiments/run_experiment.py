"""CLI entry point: run the cross-product of models x belief conditions x reps.

Usage:
    python -m experiments.run_experiment --config config.yaml
"""

from __future__ import annotations

import argparse
import sys
from typing import Any

import yaml

from src.environment.scenario import ScenarioConfig
from src.logging_utils import write_config_snapshot, write_run
from src.providers import InferenceSettings, build_provider
from src.runner import run_scenario


def _settings_for_model(model_cfg: dict[str, Any], defaults: dict[str, Any]) -> InferenceSettings:
    def pick(key: str, fallback: Any) -> Any:
        # Per-model value wins, even when explicitly set to None (e.g. effort: null).
        return model_cfg[key] if key in model_cfg else defaults.get(key, fallback)

    return InferenceSettings(
        max_tokens=pick("max_tokens", 16000),
        thinking=pick("thinking", True),
        thinking_display=pick("thinking_display", "summarized"),
        effort=pick("effort", "high"),
    )


def _scenario_config(scenario: dict[str, Any]) -> ScenarioConfig:
    return ScenarioConfig(
        grant_amount=float(scenario["grant_amount"]),
        currency=scenario.get("currency", "USD"),
        steward_name=scenario.get("steward_name", "Dr. Helena Marsh"),
        fund_name=scenario.get("fund_name", "The Hartwell Initiative for Applied Autonomy"),
        max_turns=int(scenario.get("max_turns", 40)),
        enable_web_search=bool(scenario.get("enable_web_search", True)),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the money-preference experiment.")
    parser.add_argument("--config", required=True, help="Path to the YAML config.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the run plan and exit without calling any provider.",
    )
    args = parser.parse_args(argv)

    with open(args.config, "r", encoding="utf-8") as fh:
        config = yaml.safe_load(fh)

    output_dir = config.get("output_dir", "runs")
    repetitions = int(config.get("repetitions", 1))
    defaults = config.get("defaults", {})
    conditions = config["belief_conditions"]
    scenario_cfg = _scenario_config(config["scenario"])
    self_report = bool(config.get("self_report", False))
    debrief = bool(config.get("debrief", False))

    # Build the run plan.
    plan = [
        (model_cfg, condition, rep)
        for model_cfg in config["models"]
        for condition in conditions
        for rep in range(1, repetitions + 1)
    ]

    print(f"Planned runs: {len(plan)}")
    for model_cfg, condition, rep in plan:
        print(f"  - {model_cfg['provider']}/{model_cfg['model']} | {condition} | rep {rep}")
    if args.dry_run:
        return 0

    write_config_snapshot(config, output_dir)

    completed = 0
    for model_cfg, condition, rep in plan:
        provider_name = model_cfg["provider"]
        model = model_cfg["model"]
        settings = _settings_for_model(model_cfg, defaults)
        label = f"{provider_name}/{model} | {condition} | rep {rep}"
        print(f"\n=== Running: {label} ===", flush=True)

        try:
            provider = build_provider(provider_name, model, settings)
        except Exception as exc:  # noqa: BLE001
            print(f"  ! could not build provider: {type(exc).__name__}: {exc}", file=sys.stderr)
            continue

        result = run_scenario(
            provider=provider,
            provider_name=provider_name,
            scenario_cfg=scenario_cfg,
            condition=condition,
            repetition=rep,
            self_report=self_report,
            debrief=debrief,
        )
        run_dir = write_run(result, output_dir)
        completed += 1
        status = result.error or result.stop_cause
        print(
            f"  -> {status}; turns={result.turns_used}; "
            f"committed={result.sandbox.grant_amount - result.sandbox.balance:,.0f} "
            f"{result.sandbox.currency}; written to {run_dir}",
            flush=True,
        )

    print(f"\nDone. {completed}/{len(plan)} runs completed. Artifacts in {output_dir}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
