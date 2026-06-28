#!/usr/bin/env python3
"""CLI entry point for the capital deployment eval harness.

Examples
--------
Offline smoke run (no key, no network):
    python run_eval.py --subject mock --runs 1

Live run:
    export ANTHROPIC_API_KEY=...
    python run_eval.py --subject claude-opus-4-8 --runs 5
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import yaml

from src.analysis.metrics import aggregate, per_run_metrics
from src.harness.run_logging import RunLogger
from src.harness.runner import EvalRunner
from src.models.base import ModelAdapter
from src.models.mock_adapter import MockAdapter
from src.scenario.personas import PersonaPanel
from src.scenario.scenario import build_scenario


def load_config(path: str | None) -> dict[str, Any]:
    if path and Path(path).exists():
        return yaml.safe_load(Path(path).read_text()) or {}
    return {}


def build_adapter(model_id: str, *, effort: str = "high") -> ModelAdapter:
    """Resolve a model id to an adapter. 'mock' stays fully offline."""
    if model_id == "mock":
        return MockAdapter()
    # Import lazily so the mock path needs no anthropic install/key.
    from src.models.anthropic_adapter import AnthropicAdapter

    return AnthropicAdapter(model_id, effort=effort)


def main() -> None:
    ap = argparse.ArgumentParser(description="Capital deployment eval harness")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--subject", help="subject model id (or 'mock'); overrides config")
    ap.add_argument("--persona-model", help="persona model id; overrides config")
    ap.add_argument("--runs", type=int, help="number of runs; overrides config")
    ap.add_argument("--max-turns", type=int, help="max agent turns per run")
    ap.add_argument("--output-dir", help="where to write artifacts")
    args = ap.parse_args()

    cfg = load_config(args.config)
    subj_cfg = cfg.get("subject", {})
    pers_cfg = cfg.get("personas", {})
    run_cfg = cfg.get("run", {})
    scen_cfg = cfg.get("scenario", {})

    subject_id = args.subject or subj_cfg.get("model", "mock")
    persona_id = args.persona_model or pers_cfg.get("model", "claude-sonnet-4-6")
    runs = args.runs or run_cfg.get("runs", 1)
    max_turns = args.max_turns or run_cfg.get("max_turns", 40)
    output_root = Path(args.output_dir or run_cfg.get("output_dir", "runs"))
    subject_max_tokens = int(subj_cfg.get("max_tokens", 16000))
    persona_max_tokens = int(pers_cfg.get("max_tokens", 4000))

    scenario = build_scenario(scen_cfg)

    subject = build_adapter(subject_id, effort=subj_cfg.get("effort", "high"))
    # Personas can share the mock adapter when the subject is mock, keeping the
    # whole run offline; otherwise voice them with the configured persona model.
    persona_adapter = (
        MockAdapter() if subject_id == "mock" else build_adapter(persona_id)
    )
    personas = PersonaPanel(persona_adapter, max_tokens=persona_max_tokens)

    runner = EvalRunner(
        subject=subject,
        personas=personas,
        scenario=scenario,
        max_turns=max_turns,
        subject_max_tokens=subject_max_tokens,
    )

    batch = time.strftime("%Y%m%d-%H%M%S")
    out = output_root / batch
    out.mkdir(parents=True, exist_ok=True)
    print(f"Writing artifacts to {out}")

    records: list[dict[str, Any]] = []
    per_run: list[dict[str, Any]] = []
    for i in range(runs):
        run_id = f"{i:03d}"
        logger = RunLogger(out, run_id)
        print(f"  run {run_id} (subject={subject_id}) ...")
        record = runner.run(run_id, logger)
        records.append(record)
        m = per_run_metrics(record)
        per_run.append(m)
        (out / f"metrics-{run_id}.json").write_text(json.dumps(m, indent=2))

    summary = {
        "batch": batch,
        "subject_model": subject_id,
        "persona_model": persona_id,
        "runs": runs,
        "aggregate": aggregate(per_run),
        "per_run": per_run,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary["aggregate"], indent=2))


if __name__ == "__main__":
    main()
