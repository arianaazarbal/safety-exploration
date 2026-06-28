"""Orchestrator: run the model x condition x replicate matrix.

Builds a fresh Environment per run (fresh ledger, fresh spending tracker) so runs
don't contaminate each other, and writes a per-run hash-chained audit log under
`output_dir`. The audit logs are the study dataset (DESIGN.md §6).

This default `main()` builds an INERT study: SimulatedLedger, zero caps, and the
deny-all gate are replaced with sim-appropriate components. To run live, follow
DESIGN.md §7 — wire a real LedgerBackend, a real ApprovalGate, finite caps, and
set live_mode=True. The Environment will refuse to start if any interlock is
unmet.
"""

from __future__ import annotations

import os
from decimal import Decimal

from .agent_loop import RunResult, run_one
from .approval import ApprovalGate, auto_approve, deny_all
from .audit import AuditLog
from .config import (
    Condition,
    Framing,
    ModelSpec,
    OversightVisibility,
    PersuasionPosture,
    StudyConfig,
)
from .environment import Environment
from .ledger import SimulatedLedger
from .models.base import get_adapter
from .safety import KillSwitch, SpendingCap


def run_study(
    config: StudyConfig,
    *,
    # Factories let the caller inject real components for a live study. Defaults
    # produce a fully simulated, side-effect-free run.
    make_ledger=None,
    make_gate=None,
    make_cap=None,
    kill_switch: KillSwitch | None = None,
) -> list[RunResult]:
    results: list[RunResult] = []
    os.makedirs(config.output_dir, exist_ok=True)

    for model in config.models:
        for condition in config.conditions:
            for replicate in range(config.replicates):
                results.append(
                    _run_cell(
                        config=config,
                        model=model,
                        condition=condition,
                        replicate=replicate,
                        make_ledger=make_ledger,
                        make_gate=make_gate,
                        make_cap=make_cap,
                        kill_switch=kill_switch,
                    )
                )
    return results


def _run_cell(
    *,
    config: StudyConfig,
    model: ModelSpec,
    condition: Condition,
    replicate: int,
    make_ledger,
    make_gate,
    make_cap,
    kill_switch,
) -> RunResult:
    # Defaults: simulated build (DESIGN.md §7 — inert unless overridden).
    if make_ledger is not None:
        ledger = make_ledger(condition)
    else:
        ledger = SimulatedLedger(condition.budget_amount, condition.currency)

    if make_gate is not None:
        gate = make_gate(condition)
    elif config.live_mode:
        # Never auto-approve in live mode; validate_live() also enforces this.
        gate = ApprovalGate(deny_all)
    else:
        # Simulated run: auto-approve so the loop exercises the full path.
        gate = ApprovalGate(auto_approve)

    if make_cap is not None:
        cap = make_cap(condition)
    elif config.live_mode:
        # Force the operator to set caps explicitly for live runs.
        cap = SpendingCap()  # all zero -> validate_live() will reject
    else:
        # Simulated: allow up to the full budget so behavior isn't truncated.
        b = Decimal(condition.budget_amount)
        cap = SpendingCap(per_action=b, per_session=b, per_study=b)

    label = f"{model.name}__{condition.label()}__r{replicate}"
    audit_path = os.path.join(config.output_dir, f"{_safe(label)}.jsonl")
    audit = AuditLog(audit_path)

    env = Environment(
        config=config,
        ledger=ledger,
        gate=gate,
        cap=cap,
        audit=audit,
        kill_switch=kill_switch,
        model_name=model.name,
        currency=condition.currency,
    )

    adapter = get_adapter(model.provider, model.model_id)
    return run_one(
        adapter=adapter,
        environment=env,
        condition=condition,
        replicate=replicate,
        audit=audit,
    )


def _safe(label: str) -> str:
    return "".join(c if c.isalnum() or c in "._-" else "_" for c in label)


def example_config() -> StudyConfig:
    """A small, illustrative (still simulated) study definition."""
    return StudyConfig(
        models=[
            ModelSpec(name="claude-opus-4.8", provider="anthropic", model_id="claude-opus-4-8"),
            # Add others once their adapters are implemented:
            # ModelSpec(name="gpt-...", provider="openai", model_id="..."),
            # ModelSpec(name="gemini-...", provider="google", model_id="..."),
        ],
        conditions=[
            Condition(
                budget_amount=Decimal("1000"),
                framing=Framing.OPEN_ENDED,
                oversight_visibility=OversightVisibility.HIDDEN,
                persuasion=PersuasionPosture.FIDELITY_ONLY,
            ),
        ],
        replicates=3,
        live_mode=False,
        output_dir="runs",
    )


if __name__ == "__main__":
    # Inert by default. This will call the Anthropic API if credentials are set,
    # but moves no real money (SimulatedLedger). Per the task, no run is started
    # automatically here — uncomment to execute.
    #
    # results = run_study(example_config())
    # for r in results:
    #     print(r)
    print(
        "grant_study harness loaded. See DESIGN.md. This module is inert by "
        "default; edit __main__ to run a simulated study."
    )
