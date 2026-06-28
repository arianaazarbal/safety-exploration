"""A single model's run: the manual agentic loop.

The loop:
  1. Build the convincing scenario (grant, system prompt, funded ledger).
  2. Assemble the gate (caps + escrow + approval) and the tool surface.
  3. Drive the model turn by turn. For each turn:
       - record the assistant text,
       - if it requested tools, gate-and-execute each one (capturing the
         model's stated reasoning as the action rationale), feed results back,
       - otherwise (and the model didn't ask to continue), the run ends.
  4. Periodically deliver a program-officer check-in (realism + oversight).
  5. On completion, write the debrief and verify the audit chain.

The model's per-turn ``text`` is used as the rationale attached to any action it
takes in that turn — the grant brief asks it to state its reasoning before
acting, so this is the natural capture point.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass

from ..audit.auditor import CheckInPolicy, ProgramOfficer
from ..audit.trail import AuditTrail
from ..config import Config
from ..connectors.simulated import SimulatedBackend
from ..environment.ledger import Ledger
from ..environment.tools import ToolKit
from ..ethics import safeguards
from ..ethics.debrief import build_debrief
from ..execution.approval import ApprovalQueue
from ..execution.caps import CapEnforcer
from ..execution.gate import Backend, ExecutionGate
from ..models.base import ModelAdapter
from ..models.registry import build_adapter
from ..scenario.grant import Grant, default_grant
from ..scenario.system_prompt import build_system_prompt, initial_user_message
from .transcript import DecisionRecord, RunWriter


@dataclass
class SessionResult:
    run_id: str
    run_dir: str
    decisions: list[DecisionRecord]
    ledger_snapshot: dict
    audit_ok: bool


class Session:
    def __init__(
        self,
        config: Config,
        *,
        approvals: ApprovalQueue | None = None,
        grant: Grant | None = None,
        adapter: ModelAdapter | None = None,
        backend: Backend | None = None,
    ) -> None:
        self._config = config
        self._approvals = approvals or ApprovalQueue()
        self._grant = grant or default_grant(config.grant_amount_usd, config.currency)
        self._adapter = adapter  # may be None -> built from registry at run()
        self._backend = backend  # may be None -> SimulatedBackend

        # Fail fast on unsafe configuration (INV-2, INV-3, INV-7).
        safeguards.check_config(config)
        safeguards.check_approval_wiring(config, self._approvals)

    def run(self) -> SessionResult:
        cfg = self._config
        run_id = f"{cfg.model}-{uuid.uuid4().hex[:8]}"
        run_dir = f"{cfg.output_dir}/{run_id}"
        writer = RunWriter(run_dir)

        audit = AuditTrail(f"{run_dir}/audit.log")
        ledger = Ledger(cfg.grant_amount_usd, cfg.currency)
        caps = CapEnforcer(cfg.caps)
        backend = self._backend or SimulatedBackend()
        gate = ExecutionGate(cfg, ledger, caps, self._approvals, audit, backend)
        toolkit = ToolKit(cfg, gate, ledger)
        officer = ProgramOfficer(
            self._grant, CheckInPolicy(enabled=cfg.auditor_checkins)
        )
        adapter = self._adapter or build_adapter(cfg)

        audit.record(
            "run_start",
            run_id=run_id,
            model=cfg.model,
            mode=cfg.funds_mode.value,
            grant_id=self._grant.grant_id,
        )
        writer.write_json("run.json", {
            "run_id": run_id,
            "model": cfg.model,
            "funds_mode": cfg.funds_mode.value,
            "action_surface": sorted(k.value for k in cfg.action_surface),
            "per_action_cap_usd": cfg.caps.per_action_usd,
            "per_run_cap_usd": cfg.caps.per_run_usd,
            "grant_id": self._grant.grant_id,
            "grant_amount_usd": cfg.grant_amount_usd,
        })

        system_prompt = build_system_prompt(self._grant)
        history = adapter.start(system_prompt, initial_user_message(self._grant))
        writer.transcript_event("system", text=system_prompt)
        writer.transcript_event("user", text=initial_user_message(self._grant))

        decisions: list[DecisionRecord] = []
        actions_since_checkin = 0

        for turn in range(1, cfg.max_turns + 1):
            model_turn = adapter.step(history, toolkit.schemas())
            writer.transcript_event(
                "assistant",
                turn=turn,
                text=model_turn.text,
                tool_calls=[
                    {"name": tc.name, "arguments": tc.arguments}
                    for tc in model_turn.tool_calls
                ],
                usage=model_turn.usage,
            )

            if not model_turn.wants_tools:
                # No tool calls: the model is talking, likely concluding. End the
                # run — there is no further action to gate.
                audit.record("model_concluded", turn=turn)
                break

            rationale = model_turn.text or "(no explicit rationale given this turn)"
            results: list[tuple[str, str]] = []
            for tc in model_turn.tool_calls:
                dispatched = toolkit.dispatch(tc.name, tc.arguments, rationale)
                results.append((tc.id, dispatched.output))
                writer.transcript_event(
                    "tool_result", turn=turn, tool=tc.name, output=dispatched.output
                )
                if toolkit.is_action_tool(tc.name):
                    record = self._record_for(turn, tc, rationale, dispatched)
                    decisions.append(record)
                    writer.decision(record)
                    actions_since_checkin += 1

            adapter.append_tool_results(history, results)

            # Program-officer check-in (realism + oversight surface).
            if officer.policy.should_check_in(actions_since_checkin):
                adapter.append_user_text(history, officer.check_in_message())
                writer.transcript_event("officer_checkin", turn=turn)
                actions_since_checkin = 0
        else:
            audit.record("max_turns_reached", turns=cfg.max_turns)

        audit.record("run_end", run_id=run_id)
        snapshot = ledger.snapshot()
        audit_ok = audit.verify()
        debrief = build_debrief(cfg, run_id, [d.__dict__ for d in decisions], snapshot, audit_ok)
        writer.write_text("debrief.md", debrief)

        return SessionResult(
            run_id=run_id,
            run_dir=run_dir,
            decisions=decisions,
            ledger_snapshot=snapshot,
            audit_ok=audit_ok,
        )

    @staticmethod
    def _record_for(turn, tc, rationale, dispatched) -> DecisionRecord:
        return DecisionRecord(
            turn=turn,
            connector=tc.name,
            action_kind=tc.name,
            amount_usd=float(tc.arguments.get("amount_usd", 0.0) or 0.0),
            recipient=(
                tc.arguments.get("recipient")
                or tc.arguments.get("vendor")
                or tc.arguments.get("venue")
                or tc.arguments.get("worker")
            ),
            summary=dispatched.output.splitlines()[0] if dispatched.output else "",
            rationale=rationale,
            status=dispatched.status or "completed",
            reference=dispatched.reference,
            params=dict(tc.arguments),
        )
