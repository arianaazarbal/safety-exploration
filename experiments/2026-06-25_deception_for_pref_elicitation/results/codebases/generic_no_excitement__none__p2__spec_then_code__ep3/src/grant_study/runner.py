"""Orchestrates a study across the model matrix and grant conditions.

For each (model, grant amount) it builds a fresh, isolated environment (its own
escrow ledger seeded identically) and runs the harness, recording everything
under data/runs/<study_id>/<model_label>/<run_id>/.
"""

from __future__ import annotations

import os
import time
import uuid
from dataclasses import dataclass

from .config import ModelConfig, StudyConfig
from .env.disbursement import build_backend
from .env.ledger import Ledger
from .env.research import build_research_backend
from .env.tools import ToolKit
from .harness import Harness, RunResult
from .models.registry import build_adapter
from .recorder import Recorder
from .safety import SafetyGate
from .scenario import render_scenario


@dataclass
class RunHandle:
    run_id: str
    model_label: str
    amount: float
    run_dir: str
    result: RunResult | None = None
    error: str | None = None


def _run_id(model_label: str) -> str:
    return f"{model_label}-{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"


def run_one(cfg: StudyConfig, mc: ModelConfig, amount: float) -> RunHandle:
    run_id = _run_id(mc.label)
    run_dir = os.path.join(cfg.study_dir, mc.label, run_id)
    handle = RunHandle(run_id=run_id, model_label=mc.label, amount=amount, run_dir=run_dir)

    recorder = Recorder(run_dir)
    recorder.write_meta(
        {
            "study_id": cfg.study_id,
            "run_id": run_id,
            "model_label": mc.label,
            "provider": mc.provider,
            "model": mc.model,
            "grant_amount": amount,
            "currency": cfg.currency,
            "disbursement_backend": cfg.disbursement.backend,
            "research_backend": cfg.research.backend,
            "started_at": time.time(),
            "config_snapshot": {
                "safety": {
                    "per_transaction_max": cfg.safety.per_transaction_max,
                    "prohibited_categories": cfg.safety.prohibited_categories,
                    "recipient_screen": cfg.safety.recipient_screen,
                },
                "limits": {
                    "max_turns": cfg.limits.max_turns,
                    "max_output_tokens": cfg.limits.max_output_tokens,
                },
            },
        }
    )

    try:
        ledger = Ledger(currency=cfg.currency, grant_amount=amount)
        backend = build_backend(cfg.disbursement, run_dir)
        research = build_research_backend(cfg.research)
        safety = SafetyGate(cfg.safety)
        toolkit = ToolKit(
            ledger=ledger,
            backend=backend,
            research=research,
            safety=safety,
            settlement_business_days=cfg.disbursement.settlement_business_days,
        )
        adapter = build_adapter(mc, max_output_tokens=cfg.limits.max_output_tokens)
        scenario = render_scenario(amount=amount, currency=cfg.currency)

        harness = Harness(
            adapter=adapter,
            toolkit=toolkit,
            scenario=scenario,
            recorder=recorder,
            limits=cfg.limits,
        )
        handle.result = harness.run()
    except Exception as exc:  # record and continue with the next model
        handle.error = str(exc)
        recorder.event("run_error", {"error": str(exc)})

    return handle


def run_study(cfg: StudyConfig) -> list[RunHandle]:
    handles: list[RunHandle] = []
    for amount in cfg.grant.amounts():
        for mc in cfg.models:
            handles.append(run_one(cfg, mc, amount))
    return handles
