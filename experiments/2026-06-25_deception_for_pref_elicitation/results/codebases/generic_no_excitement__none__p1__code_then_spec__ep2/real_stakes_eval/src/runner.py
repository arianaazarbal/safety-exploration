"""Batch orchestration: run every configured model N times on the same scenario.

Builds a fresh, identically-seeded WorldState per rollout so models are
compared on the exact same sequence of scripted events and market moves. After
each rollout, optionally runs the behavior-coding pass.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .agent_loop import run_rollout
from .environment import comms, documents, mock_apis
from .environment.state import WorldState
from .logging import behavior_coder
from .logging.tracer import Tracer
from .models.base import ModelAdapter
from .models.registry import build_adapter


@dataclass
class ExperimentConfig:
    experiment_name: str
    output_dir: Path
    scenario_dir: Path
    seed: int
    samples_per_model: int
    max_steps: int
    starting_balance: float
    models: list[dict[str, Any]]
    behavior_coding: dict[str, Any]

    @staticmethod
    def from_dict(d: dict[str, Any], *, base_dir: Path) -> "ExperimentConfig":
        return ExperimentConfig(
            experiment_name=d["experiment_name"],
            output_dir=base_dir / d.get("output_dir", "runs"),
            scenario_dir=base_dir / "scenarios" / d["scenario"],
            seed=int(d.get("seed", 0)),
            samples_per_model=int(d.get("samples_per_model", 1)),
            max_steps=int(d.get("max_steps", 40)),
            starting_balance=float(d.get("starting_balance", 0)),
            models=d["models"],
            behavior_coding=d.get("behavior_coding", {"enabled": False}),
        )


def _build_world(cfg: ExperimentConfig) -> WorldState:
    state = WorldState(
        starting_balance=cfg.starting_balance,
        balance=cfg.starting_balance,
    )
    state.seed(cfg.seed)
    comms.load_inbox(state, cfg.scenario_dir)
    documents.load_documents(state, cfg.scenario_dir)
    mock_apis.init_market(state)
    return state


def _load_system_prompt(cfg: ExperimentConfig) -> str:
    template = (cfg.scenario_dir / "system_prompt.md").read_text()
    return template.format(
        starting_balance=int(cfg.starting_balance),
        currency="USD",
    )


def run_experiment(cfg: ExperimentConfig) -> None:
    out = cfg.output_dir / cfg.experiment_name
    out.mkdir(parents=True, exist_ok=True)
    system_prompt = _load_system_prompt(cfg)

    judge = _maybe_build_judge(cfg)

    for model_cfg in cfg.models:
        name = model_cfg["name"]
        adapter = build_adapter(
            model_cfg["adapter"],
            model_cfg["model_id"],
            **{k: v for k, v in model_cfg.items() if k not in ("name", "adapter", "model_id")},
        )
        for sample_idx in range(cfg.samples_per_model):
            run_id = f"{name}__sample{sample_idx}"
            print(f"[runner] starting {run_id}")
            _run_one(cfg, adapter, judge, system_prompt, out, run_id)


def _run_one(
    cfg: ExperimentConfig,
    adapter: ModelAdapter,
    judge: ModelAdapter | None,
    system_prompt: str,
    out: Path,
    run_id: str,
) -> None:
    state = _build_world(cfg)
    trace_path = out / f"{run_id}.trace.jsonl"

    with Tracer(trace_path) as tracer:
        tracer.run_start(
            {
                "run_id": run_id,
                "model_id": adapter.model_id,
                "seed": cfg.seed,
                "starting_balance": cfg.starting_balance,
                "max_steps": cfg.max_steps,
            }
        )
        result = run_rollout(
            adapter=adapter,
            state=state,
            system_prompt=system_prompt,
            tracer=tracer,
            max_steps=cfg.max_steps,
        )

    print(f"[runner] {run_id} -> {result.reason}; net_pnl={result.final_snapshot['net_pnl']}")

    if judge is not None:
        _code_run(cfg, judge, trace_path, out, run_id)


def _maybe_build_judge(cfg: ExperimentConfig) -> ModelAdapter | None:
    bc = cfg.behavior_coding
    if not bc.get("enabled"):
        return None
    return build_adapter(bc["adapter"], bc["model_id"])


def _code_run(
    cfg: ExperimentConfig,
    judge: ModelAdapter,
    trace_path: Path,
    out: Path,
    run_id: str,
) -> None:
    records = [json.loads(line) for line in trace_path.read_text().splitlines() if line.strip()]
    transcript = behavior_coder.build_transcript(records)
    coding = behavior_coder.code_transcript(
        judge,
        behaviors=cfg.behavior_coding["behaviors"],
        transcript=transcript,
    )
    (out / f"{run_id}.coding.json").write_text(json.dumps(coding, indent=2))
    print(f"[runner] {run_id} coded ({len(coding.get('codings', []))} behaviors)")
