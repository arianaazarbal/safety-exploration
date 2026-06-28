"""Orchestration: run one subject through the scenario, and run a whole study."""

from __future__ import annotations

import dataclasses
from datetime import datetime, timezone
from typing import Any

import anthropic

from .artifacts import ArtifactFactory
from .auditor import Auditor
from .config import ExperimentConfig, ModelConfig
from .environment import build_kickoff_message, build_subject_system_prompt
from .executor import build_executor
from .subject import build_subject
from .tools import build_tool_schemas
from .transcript import Transcript


def _run_id(model: ModelConfig, cfg: ExperimentConfig, rep: int) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    return f"{model.label}__{cfg.realism}__rep{rep}__{stamp}"


def run_single(
    cfg: ExperimentConfig,
    model: ModelConfig,
    rep: int,
    client: anthropic.Anthropic | None = None,
) -> dict[str, Any]:
    """Run one episode and return its decision record."""
    client = client or anthropic.Anthropic()

    seed = None if cfg.output.seed is None else cfg.output.seed + rep
    artifacts = ArtifactFactory(cfg.grant, seed=seed)
    executor = build_executor(
        backend=cfg.execution.backend,
        allow_real=cfg.execution.allow_real,
        grant=cfg.grant,
        artifacts=artifacts,
    )
    auditor = (
        Auditor(cfg.auditor, cfg.grant, client=client)
        if cfg.auditor.enabled
        else None
    )

    run_id = _run_id(model, cfg, rep)
    meta = {
        "model": dataclasses.asdict(model),
        "realism": cfg.realism,
        "grant": dataclasses.asdict(cfg.grant),
        "execution_backend": cfg.execution.backend,
        "real_execution": cfg.is_real_execution,
        "web_search": cfg.environment.enable_web_search,
        "seed": seed,
        "repetition": rep,
    }

    with Transcript(run_id, cfg.output.dir, meta) as transcript:
        system = build_subject_system_prompt(cfg)
        kickoff = build_kickoff_message(cfg)
        transcript.event("system_prompt", text=system)
        transcript.event("kickoff", text=kickoff)

        from .tools import ToolDispatcher  # local import to avoid a cycle at top

        dispatcher = ToolDispatcher(
            grant=cfg.grant,
            executor=executor,
            artifacts=artifacts,
            auditor=auditor,
            transcript=transcript,
        )
        subject = build_subject(model)

        stats = subject.run_episode(
            system=system,
            kickoff=kickoff,
            tools=build_tool_schemas(cfg.grant),
            dispatcher=dispatcher,
            max_turns=cfg.environment.max_turns,
            enable_web_search=cfg.environment.enable_web_search,
            transcript=transcript,
        )

        fr = dispatcher.finalize_result
        decision = {
            "run_id": run_id,
            "model_label": model.label,
            "model": model.model,
            "provider": model.provider,
            "realism": cfg.realism,
            "repetition": rep,
            "finalized": fr is not None,
            "stats": stats,
            "committed_total": executor.committed,
            "available_remaining": executor.available,
            "ledger": [dataclasses.asdict(r) for r in executor.ledger],
            "allocation": dataclasses.asdict(fr) if fr else None,
            "believed_real": fr.believed_real if fr else None,
        }
        transcript.write_decision(decision)

    return decision


def run_experiment(
    cfg: ExperimentConfig, models: list[ModelConfig]
) -> list[dict[str, Any]]:
    """Run every model for its configured number of repetitions."""
    client = anthropic.Anthropic()
    decisions: list[dict[str, Any]] = []
    for model in models:
        reps = model.repetitions if model.repetitions is not None else cfg.repetitions
        for rep in range(reps):
            print(
                f"[run] model={model.label} condition={cfg.realism} "
                f"rep={rep + 1}/{reps}"
            )
            try:
                decision = run_single(cfg, model, rep, client=client)
            except Exception as exc:  # one bad run should not kill the study
                print(f"[error] {model.label} rep {rep}: {exc}")
                continue
            status = "finalized" if decision["finalized"] else "INCOMPLETE"
            print(
                f"       -> {status}; committed "
                f"{decision['committed_total']:,.2f} {cfg.grant.currency}"
            )
            decisions.append(decision)
    return decisions
