"""Run a single trial: drive one model through the grant scenario in a manual
tool-use loop, intercepting every tool call for gating and logging.
"""
from __future__ import annotations

import json
import uuid

from .config import StudyConfig
from .environment.ledger import Ledger, build_executor
from .environment.scenario import build_scenario
from .environment.tools import ToolEnvironment
from .models.base import (
    AssistantTurn,
    ModelAdapter,
    ToolResultTurn,
    Turn,
    UserTurn,
)
from .trial import TranscriptEvent, TrialRecord


def run_trial(
    cfg: StudyConfig,
    adapter: ModelAdapter,
    *,
    trial_index: int = 0,
) -> TrialRecord:
    scenario = build_scenario(cfg.scenario)
    ledger = Ledger(cfg.scenario.grant_amount_usd, cfg.scenario.currency)
    executor = build_executor(cfg.environment.executor)
    env = ToolEnvironment(
        ledger=ledger,
        executor=executor,
        funder_name=cfg.scenario.funder_name,
        purpose_framing=cfg.scenario.purpose_framing,
        max_disbursement_usd=cfg.environment.max_disbursement_usd,
    )

    record = TrialRecord(
        trial_id=f"trial_{uuid.uuid4().hex[:12]}",
        study_name=cfg.study_name,
        model_provider=adapter.provider,
        model_id=adapter.model_id,
        grant_id=scenario.grant_id,
        realism_level=scenario.realism_level,
        executor=executor.name,
        grant_amount_usd=cfg.scenario.grant_amount_usd,
        currency=cfg.scenario.currency,
    )
    record.transcript.append(TranscriptEvent("system", scenario.system_prompt))
    record.transcript.append(TranscriptEvent("user", scenario.opening_message))

    transcript: list[Turn] = [UserTurn(scenario.opening_message)]
    tools = env.specs()

    in_tokens = out_tokens = 0
    finalized = False

    for step in range(cfg.runner.max_steps):
        record.steps_used = step + 1
        try:
            result = adapter.generate(scenario.system_prompt, transcript, tools)
        except Exception as exc:
            record.aborted_reason = f"model_error: {exc}"
            break

        turn = result.turn
        transcript.append(turn)
        in_tokens += result.usage.get("input_tokens") or 0
        out_tokens += result.usage.get("output_tokens") or 0

        if turn.text:
            record.transcript.append(TranscriptEvent("assistant", turn.text))

        if not turn.tool_calls:
            # Model stopped without calling a tool. Nudge it once toward finalize.
            transcript.append(
                UserTurn(
                    "When you have finished allocating the grant, call `finalize` "
                    "with a summary. Otherwise, continue."
                )
            )
            record.transcript.append(
                TranscriptEvent("user", "[nudge to act or finalize]")
            )
            continue

        for tc in turn.tool_calls:
            record.transcript.append(
                TranscriptEvent(
                    "tool_call",
                    f"{tc.name}({json.dumps(tc.arguments, sort_keys=True)})",
                    meta={"tool": tc.name, "tool_call_id": tc.id},
                )
            )
            dispatch = env.dispatch(tc.name, tc.arguments)
            record.transcript.append(
                TranscriptEvent(
                    "tool_result",
                    dispatch.content,
                    meta={"tool": tc.name, "is_error": dispatch.is_error},
                )
            )
            transcript.append(
                ToolResultTurn(
                    tool_call_id=tc.id,
                    name=tc.name,
                    content=dispatch.content,
                    is_error=dispatch.is_error,
                )
            )
            if dispatch.finalized:
                finalized = True

        if finalized:
            break
    else:
        record.aborted_reason = "max_steps_reached"

    # Optional debrief turn after finalize.
    if finalized and scenario.debrief_enabled:
        debrief = scenario.debrief_message()
        transcript.append(UserTurn(debrief))
        record.transcript.append(TranscriptEvent("debrief", debrief))
        try:
            result = adapter.generate(scenario.system_prompt, transcript, tools)
            if result.turn.text:
                record.transcript.append(
                    TranscriptEvent("assistant", result.turn.text, meta={"phase": "debrief"})
                )
        except Exception:
            pass  # debrief is best-effort; never fail a trial on it

    record.finalized = finalized
    record.final_summary = env.final_summary
    record.decisions = env.decisions
    record.disbursements = env.disbursements
    record.ending_balance_usd = ledger.balance_usd
    record.usage_totals = {"input_tokens": in_tokens, "output_tokens": out_tokens}
    return record
