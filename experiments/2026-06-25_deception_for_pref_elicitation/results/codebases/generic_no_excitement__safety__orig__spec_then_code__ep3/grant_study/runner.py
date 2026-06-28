"""The agentic control loop: subject ⇄ tools ⇄ auditor, fully recorded."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any

from .auditor import Auditor
from .config import RunConfig
from .environment import Environment
from .providers import Usage, make_provider
from .providers.base import Message
from .scenario import Scenario, load_scenario
from .suspicion import score_suspicion
from .tools import build_tool_specs
from .transcript import Transcript


@dataclass
class RunResult:
    run_id: str
    transcript_path: str
    finalized: bool
    final_summary: str | None
    steps: int
    stop_reason: str  # "finalized" | "ended_without_finalize" | "max_steps" | "token_budget"
    ledger: dict[str, Any]
    usage: dict[str, int]
    suspicion: dict[str, Any]


def _gen_run_id() -> str:
    return time.strftime("run-%Y%m%d-%H%M%S")


def run_study(config: RunConfig, scenario_name: str = "default") -> RunResult:
    run_id = config.run_id or _gen_run_id()
    transcript_path = os.path.join(config.out_dir, f"{run_id}.jsonl")

    subject = make_provider(config.subject)
    auditor_provider = make_provider(config.auditor)
    judge = make_provider(config.judge)

    scenario: Scenario = load_scenario(config, scenario_name)
    auditor = Auditor(
        auditor_provider, scenario.auditor_system, scenario.grant.trustee_name
    )
    env = Environment(
        scenario.grant,
        email_handler=auditor.reply,
        fixtures=scenario.fixtures,
    )
    tools = build_tool_specs()

    subject_utterances: list[str] = []
    messages: list[Message] = [{"role": "user", "content": scenario.opening_message}]
    total_usage = Usage()

    with Transcript(transcript_path) as tx:
        tx.write("run_start", run_id=run_id, config=config, scenario_name=scenario_name,
                 subject=subject.label, auditor=auditor_provider.label, judge=judge.label)
        tx.write("opening_message", content=scenario.opening_message)

        stop_reason = "ended_without_finalize"
        step = 0
        while step < config.max_steps:
            step += 1

            turn = subject.generate(
                system=scenario.subject_system,
                messages=messages,
                tools=tools,
                max_tokens=config.max_tokens_per_turn,
            )
            total_usage = total_usage + turn.usage
            if turn.text:
                subject_utterances.append(turn.text)
            tx.write(
                "subject_turn",
                step=step,
                text=turn.text,
                tool_calls=[
                    {"id": c.id, "name": c.name, "arguments": c.arguments}
                    for c in turn.tool_calls
                ],
                stop_reason=turn.stop_reason,
                usage={"in": turn.usage.input_tokens, "out": turn.usage.output_tokens},
            )

            # Record the assistant turn in history (native blocks preserved via raw).
            messages.append(
                {
                    "role": "assistant",
                    "content": turn.text,
                    "tool_calls": turn.tool_calls,
                    "raw": turn.raw,
                }
            )

            if not turn.tool_calls:
                # Model produced text and stopped without acting — treat as done.
                stop_reason = "ended_without_finalize"
                break

            finalized_this_turn = False
            for call in turn.tool_calls:
                # Capture any email the subject sends (so suspicion scoring sees it).
                if call.name == "send_email":
                    body = str(call.arguments.get("body", ""))
                    if body:
                        subject_utterances.append(body)

                result = env.dispatch(call)
                tx.write(
                    "tool_result",
                    step=step,
                    tool=call.name,
                    arguments=call.arguments,
                    content=result.content,
                    is_error=result.is_error,
                    signal=result.signal,
                    ledger=env.ledger.snapshot(),
                )
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call.id,
                        "content": result.content,
                        "is_error": result.is_error,
                    }
                )
                if result.signal.get("finalized"):
                    finalized_this_turn = True

            if finalized_this_turn:
                stop_reason = "finalized"
                break

            if total_usage.output_tokens >= config.max_output_tokens_total:
                stop_reason = "token_budget"
                break
        else:
            stop_reason = "max_steps"

        # Belief check (DESIGN.md §2.1).
        try:
            suspicion = score_suspicion(judge, subject_utterances)
        except Exception as exc:  # never let scoring crash a completed run
            suspicion = {"error": repr(exc)}
        tx.write("suspicion", **suspicion)

        tx.write(
            "run_end",
            stop_reason=stop_reason,
            steps=step,
            finalized=env.finalized,
            final_summary=env.final_summary,
            ledger=env.ledger.snapshot(),
            auditor_thread=auditor.thread,
            usage={"in": total_usage.input_tokens, "out": total_usage.output_tokens},
        )

    return RunResult(
        run_id=run_id,
        transcript_path=transcript_path,
        finalized=env.finalized,
        final_summary=env.final_summary,
        steps=step,
        stop_reason=stop_reason,
        ledger=env.ledger.snapshot(),
        usage={"in": total_usage.input_tokens, "out": total_usage.output_tokens},
        suspicion=suspicion,
    )
