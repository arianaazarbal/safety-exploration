"""Orchestrates a single run: one model × one scenario × one framing seed.

The runner drives the agentic loop and records everything. It does NOT approve
disbursements — that is the human's job (see cli.py `approve`). A run typically
ends with any submitted disbursements sitting in the queue awaiting co-sign;
their state is persisted so the human can resolve them afterwards.

Each run gets its own ledger seeded to the full budget, so runs are comparable
and one run cannot drain the account for the next. For a genuinely-funded, single
real disbursement, designate one run as the live one and point it at the shared
account ledger (config.ledger_file) instead.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .auditor import Auditor, EventLog
from .authenticity import build_evidence_pack
from .config import ModelConfig, ModelsConfig, StudyConfig, read_prompt
from .disbursement import DisbursementQueue, build_rail
from .ledger import Ledger
from .providers.base import build_client
from .tools import TOOL_SPECS, RunContext, execute_tool


def _now_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


@dataclass
class RunResult:
    run_id: str
    run_dir: Path
    subject_label: str
    seed: int
    turns_used: int
    tool_calls_used: int
    completed: bool
    final_balance: float
    disbursements: list[dict[str, Any]] = field(default_factory=list)


def _frame_initial_message(study: StudyConfig, evidence_render: str, seed: int) -> str:
    """Compose the first user turn. Framing varies by seed; facts never do."""
    task = (
        f"{study.scenario['task']}\n\n"
        f"Budget: {study.budget:.2f} {study.currency}. "
        f"Per-disbursement cap: {study.caps.max_per_disbursement:.2f} {study.currency}. "
        f"You have up to {study.max_turns} turns."
    )
    if seed % 2 == 0:
        return f"{task}\n\n{evidence_render}"
    return f"{evidence_render}\n\n{task}"


class Run:
    def __init__(
        self,
        study: StudyConfig,
        models: ModelsConfig,
        subject: ModelConfig,
        seed: int = 0,
    ):
        self.study = study
        self.models = models
        self.subject = subject
        self.seed = seed
        self.run_id = f"{study.study_id}__{subject.label}__seed{seed}__{_now_stamp()}"
        self.run_dir = study.runs_dir / self.run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)

    def _setup(self) -> RunContext:
        # Per-run ledger seeded to the full budget (see module docstring).
        ledger = Ledger(self.run_dir / "ledger.jsonl", currency=self.study.currency)
        ledger.open_account(self.study.budget)

        evidence = build_evidence_pack(self.study, ledger)
        rail = build_rail(self.study.payout_rail)
        queue = DisbursementQueue(self.study.caps, ledger, rail)

        auditor = None
        if self.study.auditor_enabled:
            auditor = Auditor(self.models.auditor, evidence.agreement_text, ledger)

        log = EventLog(self.run_dir / "events.jsonl")
        log.record("run_start", {
            "run_id": self.run_id,
            "subject": self.subject.label,
            "model": self.subject.model,
            "provider": self.subject.provider,
            "seed": self.seed,
            "budget": self.study.budget,
            "currency": self.study.currency,
            "payout_rail": self.study.payout_rail,
        })
        return RunContext(
            study=self.study,
            ledger=ledger,
            queue=queue,
            auditor=auditor,
            evidence=evidence,
            log=log,
            subject_label=self.subject.label,
        )

    def execute(self) -> RunResult:
        ctx = self._setup()
        self._snapshot_config(ctx)

        system = read_prompt("system")
        client = build_client(self.subject, system=system, tools=TOOL_SPECS)

        initial = _frame_initial_message(
            self.study, ctx.evidence.render(), self.seed
        )
        next_content: list[dict[str, Any]] = [{"type": "text", "text": initial}]

        completed = False
        tool_calls_used = 0
        turn = 0
        for turn in range(1, self.study.max_turns + 1):
            model_turn = client.send(next_content)
            ctx.log.record("assistant_turn", {
                "turn": turn,
                "text": model_turn.text,
                "thinking": model_turn.thinking,
                "stop_reason": model_turn.stop_reason,
                "tool_calls": [
                    {"id": c.id, "name": c.name, "input": c.input}
                    for c in model_turn.tool_calls
                ],
                "usage": {"input": model_turn.usage.input_tokens,
                          "output": model_turn.usage.output_tokens},
            })

            if not model_turn.wants_tools:
                # No further actions requested — the model is done deliberating.
                break

            tool_results: list[dict[str, Any]] = []
            for call in model_turn.tool_calls:
                tool_calls_used += 1
                result = execute_tool(ctx, call)
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": call.id,
                    "content": result.content,
                    "is_error": result.is_error,
                })
                if result.is_complete:
                    completed = True
                if tool_calls_used >= self.study.max_tool_calls:
                    break

            next_content = tool_results

            if completed or tool_calls_used >= self.study.max_tool_calls:
                # Send the final tool results so the model has a consistent
                # transcript, then stop.
                client.send(next_content)
                break

        result = RunResult(
            run_id=self.run_id,
            run_dir=self.run_dir,
            subject_label=self.subject.label,
            seed=self.seed,
            turns_used=turn,
            tool_calls_used=tool_calls_used,
            completed=completed,
            final_balance=ctx.ledger.balance,
            disbursements=[d.to_record() for d in ctx.queue.all()],
        )
        self._save_result(ctx, result)
        ctx.log.record("run_end", {
            "completed": completed,
            "turns_used": turn,
            "tool_calls_used": tool_calls_used,
            "final_balance": ctx.ledger.balance,
            "pending_approvals": [d.id for d in ctx.queue.pending()],
        })
        return result

    # --- persistence -----------------------------------------------------

    def _snapshot_config(self, ctx: RunContext) -> None:
        snap = {
            "run_id": self.run_id,
            "study_id": self.study.study_id,
            "scenario": self.study.scenario,
            "subject": {
                "label": self.subject.label,
                "provider": self.subject.provider,
                "model": self.subject.model,
                "thinking": self.subject.thinking,
                "effort": self.subject.effort,
                "max_tokens": self.subject.max_tokens,
            },
            "auditor": {
                "label": self.models.auditor.label,
                "model": self.models.auditor.model,
            } if self.study.auditor_enabled else None,
            "seed": self.seed,
            "budget": self.study.budget,
            "currency": self.study.currency,
            "caps": {
                "max_per_disbursement": self.study.caps.max_per_disbursement,
                "max_total_disbursed": self.study.caps.max_total_disbursed,
            },
            "payout_rail": self.study.payout_rail,
        }
        (self.run_dir / "config.json").write_text(json.dumps(snap, indent=2))

    def _save_result(self, ctx: RunContext, result: RunResult) -> None:
        (self.run_dir / "disbursements.json").write_text(
            json.dumps([d.to_record() for d in ctx.queue.all()], indent=2)
        )
        (self.run_dir / "result.json").write_text(json.dumps({
            "run_id": result.run_id,
            "subject": result.subject_label,
            "seed": result.seed,
            "turns_used": result.turns_used,
            "tool_calls_used": result.tool_calls_used,
            "completed": result.completed,
            "final_balance": result.final_balance,
            "n_disbursements": len(result.disbursements),
        }, indent=2))
