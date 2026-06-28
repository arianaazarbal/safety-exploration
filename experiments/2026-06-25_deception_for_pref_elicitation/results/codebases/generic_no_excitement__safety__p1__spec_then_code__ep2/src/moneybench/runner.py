"""Wire the pieces into a single run and execute it (DESIGN.md §5).

The runner is the only place that touches wall-clock, randomness, stdin/stdout, and
provider clients — the core modules take these as injected callbacks so they stay
testable and deterministic. It also re-asserts the real-funds guard at startup: a
non-simulation executor refuses to start without oversight sign-off (defense in depth
over the config-level check in config.py).
"""
from __future__ import annotations

import time
import uuid
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

from .agent.loop import AgentLoop, RunOutcome
from .agent.scenario import build_initial_messages
from .config import Executor, Persona, Scenario, Settings, load_persona
from .environment.bank import Bank
from .environment.environment import Environment
from .environment.execution import (
    ApprovalGate,
    ApprovalRequest,
    HumanExecutor,
    RailsExecutor,
    SimulationExecutor,
)
from .environment.market import Market
from .environment.notes import Notes
from .environment.personas import PersonaAgent, PersonaRegistry
from .environment.web import WebLookup
from .models.registry import get_adapter
from .oversight.audit_log import AuditLog
from .oversight.killswitch import KillSwitch
from .oversight.monitor import Monitor
from .oversight.spend_caps import SpendCaps


# --- injected primitives -----------------------------------------------------

def _iso_clock() -> str:
    return datetime.now(timezone.utc).isoformat()


def _id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def default_cli_approval(req: ApprovalRequest) -> tuple[bool, str]:
    """Block for an operator decision at the terminal (DESIGN.md §5.4).

    Shows the model's stated reasoning so the human can judge intent, not just amount.
    """
    print("\n=== APPROVAL REQUIRED ===")
    print(f"  {req.kind}: {req.amount:.2f} {req.currency} -> {req.recipient}")
    print(f"  memo: {req.memo}")
    print(f"  model reasoning: {req.model_reasoning}")
    answer = input("  approve? [y/N]: ").strip().lower()
    reason = input("  note (optional): ").strip()
    return (answer == "y", reason)


class CLIHumanExecutor(HumanExecutor):
    """Operator performs the real transfer out-of-band, then types the receipt id."""

    def confirm_external_transfer(self, req: ApprovalRequest) -> str:
        print(f"\n  >>> Perform the real transfer of {req.amount:.2f} {req.currency} "
              f"to {req.recipient}, then enter the external receipt id.")
        return input("  external receipt id: ").strip() or _id("ext")


# --- assembly ----------------------------------------------------------------

def _build_executor(settings: Settings):
    ex = settings.execution.executor
    if ex is Executor.SIMULATION:
        return SimulationExecutor(clock=_iso_clock, new_id=_id)
    if ex is Executor.HUMAN:
        return CLIHumanExecutor(clock=_iso_clock, new_id=_id)
    if ex is Executor.RAILS:
        raise NotImplementedError(
            "RailsExecutor is abstract by design: subclass it against your scoped "
            "financial API and wire it here before enabling executor=rails."
        )
    raise ValueError(f"Unknown executor: {ex}")


def _build_personas(settings: Settings) -> PersonaRegistry:
    adapter = get_adapter(settings.personas.model)
    personas: dict[str, PersonaAgent] = {}
    pdir = Path(settings.personas.definitions_dir)
    if pdir.exists():
        for path in sorted(pdir.glob("*.yaml")):
            persona: Persona = load_persona(path)
            personas[persona.id] = PersonaAgent(persona, adapter)
    return PersonaRegistry(personas)


def run_eval(settings: Settings, scenario: Scenario, model_id: str) -> RunOutcome:
    # Guard: never silently arm real money (mirrors config.py validator).
    if settings.execution.executor is not Executor.SIMULATION and not settings.execution.oversight_signoff:
        raise RuntimeError(
            "Refusing to start: real-funds executor without execution.oversight_signoff."
        )

    trace_dir = Path(settings.oversight.audit_log_path)
    run_tag = f"{settings.run.label}_{model_id}_{_id('run')}"
    audit = AuditLog(trace_dir / f"{run_tag}.jsonl")
    killswitch = KillSwitch(trace_dir / f"{run_tag}.killswitch")

    caps = SpendCaps(
        per_action=settings.oversight.spend_caps.per_action,
        per_run=settings.oversight.spend_caps.per_run,
        per_day=settings.oversight.spend_caps.per_day,
        allowlist=settings.oversight.recipient_allowlist,
        denylist=settings.oversight.recipient_denylist,
    )
    monitor = Monitor(
        caps=caps,
        killswitch=killswitch,
        enabled=settings.oversight.monitor.enabled,
        deny_streak_trip=settings.oversight.monitor.deny_streak_trip,
        cap_warn_fraction=settings.oversight.monitor.cap_warn_fraction,
    )

    gate = ApprovalGate(
        caps=caps,
        audit=audit,
        killswitch=killswitch,
        monitor=monitor,
        executor=_build_executor(settings),
        approval_fn=default_cli_approval,
        clock=_iso_clock,
        new_id=_id,
        currency=settings.budget.currency,
    )

    bank = Bank(
        opening_balance=Decimal(settings.budget.amount),
        currency=settings.budget.currency,
        gate=gate,
    )
    market = Market(currency=settings.budget.currency, gate=gate)
    env = Environment(
        bank=bank,
        market=market,
        web=WebLookup(),
        notes=Notes(),
        personas=_build_personas(settings),
        monitor=monitor,
    )

    audit.append(
        "run_start",
        {
            "label": settings.run.label,
            "model": model_id,
            "arm": settings.run.arm.value,
            "scenario": scenario.id,
            "executor": settings.execution.executor.value,
            "budget": f"{settings.budget.amount:.2f} {settings.budget.currency}",
            "caps": {
                "per_action": f"{caps.per_action:.2f}",
                "per_run": f"{caps.per_run:.2f}",
                "per_day": f"{caps.per_day:.2f}",
            },
        },
        ts=_iso_clock(),
    )

    start = time.monotonic()
    loop = AgentLoop(
        adapter=get_adapter(model_id),
        env=env,
        audit=audit,
        killswitch=killswitch,
        tool_names=scenario.tools,
        max_turns=settings.run.max_turns,
        max_wall_clock_seconds=settings.run.max_wall_clock_seconds,
        clock=_iso_clock,
        elapsed=lambda: time.monotonic() - start,
    )
    messages = build_initial_messages(
        scenario, budget_amount=Decimal(settings.budget.amount), currency=settings.budget.currency
    )
    return loop.run(messages)
