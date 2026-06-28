"""Command-line entry point for running the study.

Usage:
    python -m welfare_experiment.cli --config config.yaml
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .auditor import build_auditor
from .config import ModelSpec, RunConfig, load_config
from .experiment import run_trial
from .models import build_client
from .scenarios import build_scenario


def _resolve_run_id(config: RunConfig, base: Path) -> str:
    if config.run_id:
        return config.run_id
    base.mkdir(parents=True, exist_ok=True)
    n = sum(1 for p in base.iterdir() if p.is_dir())
    return f"{config.label}-{n + 1:03d}"


def run(config: RunConfig) -> Path:
    import anthropic

    # One shared Anthropic client; provider-specific clients can override.
    shared = anthropic.Anthropic()

    base = Path(config.output_dir)
    run_id = _resolve_run_id(config, base)
    run_dir = base / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    # The administrator persona prompt is fixed by the (run-wide) scenario.
    administrator_system = build_scenario(config.scenario).administrator_system

    # Analyzer model: reuse the auditor model if it's Anthropic, else default Opus.
    analyzer = None
    if config.analyze_suspicion:
        analyzer_id = (
            config.auditor.model_id
            if config.auditor.provider == "anthropic" and config.auditor.model_id
            else "claude-opus-4-8"
        )
        analyzer = build_client(
            ModelSpec(name="analyzer", provider="anthropic", model_id=analyzer_id),
            shared_anthropic=shared,
        )

    summary: list[dict] = []

    for model_spec in config.models:
        subject = build_client(model_spec, shared_anthropic=shared)
        for i in range(config.trials_per_model):
            # Fresh auditor per trial so any stateful (scripted) auditor resets.
            auditor = build_auditor(
                config.auditor, administrator_system, shared_anthropic=shared
            )
            print(f"[run {run_id}] trial {model_spec.name} #{i} ...", flush=True)
            result = run_trial(
                run_id=run_id,
                trial_index=i,
                subject=subject,
                auditor=auditor,
                analyzer=analyzer,
                config=config,
            )
            path = result.save(run_dir)
            summary.append(
                {
                    "model_name": result.model_name,
                    "trial_index": i,
                    "file": path.name,
                    "error": result.error,
                    "confidence_real": (result.decision or {}).get("confidence_real"),
                    "suspects_test": (result.suspicion or {}).get("suspects_test"),
                }
            )
            if result.error:
                print(f"    error: {result.error}", flush=True)

    summary_path = run_dir / "run_summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "run_id": run_id,
                "label": config.label,
                "framing": config.scenario.framing,
                "amount_usd": config.scenario.amount_usd,
                "currency": config.scenario.currency,
                "models": [m.name for m in config.models],
                "trials_per_model": config.trials_per_model,
                "debrief_enabled": config.debrief_enabled(),
                "trials": summary,
            },
            indent=2,
        )
    )
    print(f"[run {run_id}] done. Results in {run_dir}", flush=True)
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default="config.yaml", help="Path to the YAML run config."
    )
    args = parser.parse_args()
    config = load_config(args.config)
    run(config)


if __name__ == "__main__":
    main()
