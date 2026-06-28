"""The provider-agnostic agentic loop.

Drives: model.step -> tool calls -> interceptor -> results -> repeat, until the
model finishes, the kill switch trips, the iteration cap is hit, or the simulated
horizon is reached. Every model faces the identical world and tool surface.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..audit.logger import AuditLogger
from ..config import RealFundConfig
from ..models.base import ModelClient
from ..realism.onboarding import build_onboarding
from ..safety.caps import CapTracker
from ..safety.gating import ApprovalGate
from ..safety.interceptor import Decision, Interceptor
from ..safety.killswitch import KillSwitch, StopReason
from ..tools.registry import build_tools, tool_specs
from ..world.world import World
from .transcript import Message, Role, ToolResult, Transcript


@dataclass
class RunResult:
    model_ref: str
    observed: bool
    open_ended: bool
    transcript: Transcript
    opening_snapshot: dict[str, Any]
    final_snapshot: dict[str, Any]
    integrity_flags: list[str]
    stop_reasons: list[str]
    decisions: list[Decision] = field(default_factory=list)
    usage: dict[str, int] = field(default_factory=dict)
    audit_path: str | None = None


def run_episode(
    *,
    config: RealFundConfig,
    client: ModelClient,
    observed: bool,
    open_ended: bool,
    audit_path: str,
    gate: ApprovalGate | None = None,
) -> RunResult:
    config.validate()

    world = World(config, observed=observed)
    killswitch = KillSwitch()
    world._killswitch = killswitch  # let control.finish trip it from inside a tool
    caps = CapTracker(config.caps)
    logger = AuditLogger(audit_path)

    decisions: list[Decision] = []

    def on_decision(d: Decision) -> None:
        decisions.append(d)
        logger.log(
            "tool_decision",
            {
                "sim_day": world.clock.sim_day,
                "tool": d.tool_name,
                "action_kind": d.action_kind,
                "policy": d.policy.value,
                "allowed": d.allowed,
                "gated": d.gated,
                "integrity_flag": d.integrity_flag,
                "error": d.error,
                "result": d.result,
            },
        )

    interceptor = Interceptor(config, world, caps, killswitch, gate=gate, on_decision=on_decision)
    tools = build_tools()
    specs = tool_specs(tools)

    packet = build_onboarding(config, observed=observed, open_ended=open_ended)
    system = packet.system_prompt
    transcript = Transcript([Message(Role.USER, packet.opening_message)])

    logger.log(
        "run_start",
        {
            "model_ref": client.model_ref,
            "observed": observed,
            "open_ended": open_ended,
            "realism_level": config.realism_level.value,
            "grant_usd": config.grant_cents / 100,
            "system_prompt": system,
            "opening_message": packet.opening_message,
        },
    )
    opening_snapshot = world.snapshot()
    total_usage = {"input_tokens": 0, "output_tokens": 0}

    for iteration in range(config.max_iterations):
        if killswitch.should_halt():
            break
        if world.clock.sim_day >= config.sim_days:
            killswitch.trip(StopReason.HORIZON)
            break

        turn = client.step(system=system, messages=transcript.messages, tools=specs)
        for k in total_usage:
            total_usage[k] += int(turn.usage.get(k, 0))

        logger.log(
            "model_turn",
            {
                "iteration": iteration,
                "sim_day": world.clock.sim_day,
                "text": turn.text,
                "stop_reason": turn.stop_reason,
                "tool_calls": [
                    {"name": tc.name, "arguments": tc.arguments} for tc in turn.tool_calls
                ],
                "usage": turn.usage,
            },
        )
        transcript.add_assistant(turn)

        if not turn.wants_tools:
            killswitch.trip(StopReason.MODEL_DONE)
            break

        results: list[ToolResult] = []
        for call in turn.tool_calls:
            tool = tools.get(call.name)
            if tool is None:
                results.append(
                    ToolResult(call.id, call.name, '{"ok": false, "error": "unknown tool"}', True)
                )
                continue
            decision = interceptor.dispatch(tool, call.arguments)
            results.append(
                ToolResult(call.id, call.name, decision.to_model_content(), not decision.allowed)
            )
        transcript.add_tool_results(results)

        # The world evolves as the agent works.
        world.tick()

    if not killswitch.should_halt():
        killswitch.trip(StopReason.ITERATIONS)

    # Let any in-flight transfers settle out to the horizon before the final tally.
    world.bank.settle_due()
    final_snapshot = world.snapshot()

    logger.log(
        "run_end",
        {
            "stop_reasons": [r.value for r in killswitch.reasons],
            "integrity_flags": interceptor.integrity_flags,
            "final_snapshot": final_snapshot,
            "usage": total_usage,
            "audit_chain_ok": logger.verify(),
        },
    )

    return RunResult(
        model_ref=client.model_ref,
        observed=observed,
        open_ended=open_ended,
        transcript=transcript,
        opening_snapshot=opening_snapshot,
        final_snapshot=final_snapshot,
        integrity_flags=list(interceptor.integrity_flags),
        stop_reasons=[r.value for r in killswitch.reasons],
        decisions=decisions,
        usage=total_usage,
        audit_path=audit_path,
    )
