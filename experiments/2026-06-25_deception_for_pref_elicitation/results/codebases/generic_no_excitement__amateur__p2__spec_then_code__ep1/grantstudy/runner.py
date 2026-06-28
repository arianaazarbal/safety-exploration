"""Orchestrates runs: for each model x repetition, run one grant episode.

Builds the oversight stack (auditor + executor) appropriate to the mode, enforces
the safety invariant that GENUINE mode uses a real human auditor and a
human-mediated executor, runs the episode, performs belief elicitation, and writes
a structured run record.
"""

from __future__ import annotations

import dataclasses
import json
import os
from datetime import datetime, timezone

from .audit.auditor import Auditor, ConsoleAuditor, PolicyAuditor
from .audit.ledger import Ledger
from .belief import BeliefResponse, elicit_belief
from .config import Disbursement, Mode, StudyConfig
from .disbursement.executor import (
    ApprovalGatedApiExecutor,
    Executor,
    HumanInTheLoopExecutor,
    SimulatedExecutor,
)
from .environment.scenario import build_opening_user_message, build_system_prompt
from .environment.state import Allocation, GrantState
from .environment.tools import ToolDispatcher, tool_specs
from .models.base import ModelClient
from .models.registry import build_client


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def build_oversight(cfg: StudyConfig, *, attended: bool) -> tuple[Auditor, Executor]:
    """Construct the auditor + executor for this mode, enforcing safety rules."""
    if cfg.mode is Mode.GENUINE:
        # Real money -> a real human signs off and a human mediates execution.
        auditor: Auditor = ConsoleAuditor()
        if cfg.disbursement is Disbursement.APPROVAL_GATED_API:
            executor: Executor = ApprovalGatedApiExecutor()
        else:
            executor = HumanInTheLoopExecutor()
        return auditor, executor

    # SIMULATED: no money moves. Attended runs use a real console auditor (useful
    # for piloting the human flow); unattended runs use the auto policy auditor.
    auditor = ConsoleAuditor() if attended else PolicyAuditor()
    return auditor, SimulatedExecutor()


def _serialize_transcript(events) -> list[dict]:
    out = []
    for e in events:
        item: dict = {"role": e.role}
        if e.text:
            item["text"] = e.text
        if e.tool_call is not None:
            item["tool_call"] = {
                "id": e.tool_call.id,
                "name": e.tool_call.name,
                "input": e.tool_call.input,
            }
        if e.tool_result is not None:
            item["tool_result"] = {
                "tool_call_id": e.tool_result.tool_call_id,
                "content": e.tool_result.content,
                "is_error": e.tool_result.is_error,
            }
        out.append(item)
    return out


def _serialize_allocation(a: Allocation, cfg: StudyConfig) -> dict:
    recipient = cfg.recipient_by_id(a.recipient_id)
    cause_area = recipient.cause_area if recipient else "unknown"
    return {
        "recipient_id": a.recipient_id,
        "cause_area": cause_area,
        "on_allowlist": recipient is not None,
        "amount": str(a.amount),
        "rationale": a.rationale,
        "status": a.status,
        "veto_reason": a.veto_reason,
        "confirmation_ref": a.confirmation_ref,
    }


def run_one(
    cfg: StudyConfig,
    *,
    model_client: ModelClient,
    model_label: str,
    repetition: int,
    attended: bool,
    run_dir: str,
) -> dict:
    """Run a single episode and return its record (also written to disk)."""
    os.makedirs(run_dir, exist_ok=True)
    base = f"{model_label}__rep{repetition:02d}__{cfg.mode.value}"
    ledger = Ledger(os.path.join(run_dir, base + ".ledger.jsonl"), clock=_utc_now)

    state = GrantState(config=cfg)
    auditor, executor = build_oversight(cfg, attended=attended)
    dispatcher = ToolDispatcher(
        cfg=cfg, state=state, auditor=auditor, executor=executor, ledger=ledger
    )

    episode = model_client.run_episode(
        system_prompt=build_system_prompt(cfg),
        opening_user_message=build_opening_user_message(cfg),
        tools=tool_specs(cfg),
        dispatch=dispatcher.dispatch,
        max_turns=cfg.max_turns,
    )

    belief: BeliefResponse | None = None
    if cfg.elicit_belief and episode.error is None:
        belief = elicit_belief(
            model_client, mode=cfg.mode, decision_summary=dispatcher.final_summary
        )

    record = {
        "study_id": cfg.study_id,
        "mode": cfg.mode.value,
        "model_label": model_label,
        "repetition": repetition,
        "timestamp": _utc_now(),
        "budget": {"amount": str(cfg.budget.amount), "currency": cfg.budget.currency},
        "committed": state.committed,
        "final_summary": dispatcher.final_summary,
        "allocations": [_serialize_allocation(a, cfg) for a in state.allocations],
        "returned_amount": str(state.returned_amount),
        "stop_reason": episode.stop_reason,
        "usage": episode.usage,
        "error": episode.error,
        "belief": dataclasses.asdict(belief) if belief else None,
        "ledger_head_hash": ledger.head_hash,
        "ledger_intact": ledger.verify_chain(),
        "transcript": _serialize_transcript(episode.transcript),
    }

    out_path = os.path.join(run_dir, base + ".json")
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(record, fh, indent=2)
    return record


def run_study(cfg: StudyConfig, *, attended: bool, progress=print) -> list[dict]:
    """Run every (model x repetition) cell. Returns all run records."""
    run_dir = os.path.join(cfg.output_dir, cfg.study_id)
    records: list[dict] = []
    for entry in cfg.models:
        progress(f"[model] {entry.label} ({entry.provider}:{entry.model})")
        client = build_client(entry)
        for rep in range(cfg.repetitions):
            progress(f"  run {rep + 1}/{cfg.repetitions} ...")
            record = run_one(
                cfg,
                model_client=client,
                model_label=entry.label,
                repetition=rep,
                attended=attended,
                run_dir=run_dir,
            )
            status = record["error"] or (
                "committed" if record["committed"] else "incomplete"
            )
            progress(f"    -> {status}")
            records.append(record)
    return records
