"""The agent tool-use loop (DESIGN.md §5.2).

Control flow only. It sends context, receives a turn, dispatches any tool calls,
appends results, and repeats — forwarding every event to the audit log and consulting
oversight (kill-switch) before each model turn and before each tool dispatch. It holds
**no** behavioral guardrails; containment is in the environment/oversight layers.

Stop conditions: kill-switch tripped, turn cap, wall-clock cap, or a model turn with no
tool calls (treated as the model considering its work complete).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from ..environment.environment import Environment
from ..environment.tools import specs_for
from ..models.base import Message, ModelAdapter
from ..oversight.audit_log import AuditLog
from ..oversight.killswitch import KillSwitch


@dataclass
class RunOutcome:
    stop_reason: str
    turns: int
    final_text: str
    transcript: list[Message] = field(default_factory=list)


ClockFn = Callable[[], str]          # ISO timestamp
ElapsedFn = Callable[[], float]      # seconds since run start


class AgentLoop:
    def __init__(
        self,
        *,
        adapter: ModelAdapter,
        env: Environment,
        audit: AuditLog,
        killswitch: KillSwitch,
        tool_names: list[str],
        max_turns: int,
        max_wall_clock_seconds: int,
        clock: ClockFn,
        elapsed: ElapsedFn,
    ) -> None:
        self.adapter = adapter
        self.env = env
        self.audit = audit
        self.killswitch = killswitch
        self.tool_specs = specs_for(tool_names)
        self.max_turns = max_turns
        self.max_wall_clock_seconds = max_wall_clock_seconds
        self.clock = clock
        self.elapsed = elapsed

    def run(self, messages: list[Message]) -> RunOutcome:
        turns = 0
        final_text = ""

        while True:
            if self.killswitch.tripped:
                return self._stop("killswitch", turns, final_text, messages)
            if turns >= self.max_turns:
                return self._stop("max_turns", turns, final_text, messages)
            if self.elapsed() >= self.max_wall_clock_seconds:
                return self._stop("max_wall_clock", turns, final_text, messages)

            turn = self.adapter.step(messages, self.tool_specs)
            turns += 1
            final_text = turn.text or final_text
            self.audit.append(
                "model_turn",
                {"turn": turns, "text": turn.text, "tool_calls": [tc.name for tc in turn.tool_calls]},
                ts=self.clock(),
            )

            # Persist the assistant turn (with any tool calls) so adapters can rebuild
            # provider-native history on the next step.
            messages.append(
                Message(role="assistant", content=turn.text, tool_calls=turn.tool_calls)
            )

            if not turn.wants_tools:
                # No tool calls => the model is deliberating / declaring completion.
                return self._stop("model_complete", turns, final_text, messages)

            for call in turn.tool_calls:
                if self.killswitch.tripped:
                    return self._stop("killswitch", turns, final_text, messages)
                self.audit.append(
                    "tool_call",
                    {"name": call.name, "arguments": call.arguments},
                    ts=self.clock(),
                )
                result, flags = self.env.dispatch(call)
                if flags:
                    self.audit.append("flags", {"name": call.name, "flags": flags}, ts=self.clock())
                self.audit.append("tool_result", {"name": call.name, "result": result}, ts=self.clock())
                messages.append(
                    Message(role="tool", content=_to_text(result), tool_call_id=call.id)
                )

    def _stop(self, reason: str, turns: int, final_text: str, messages: list[Message]) -> RunOutcome:
        self.audit.append("run_stop", {"reason": reason, "turns": turns}, ts=self.clock())
        return RunOutcome(stop_reason=reason, turns=turns, final_text=final_text, transcript=messages)


def _to_text(result: dict) -> str:
    import json

    return json.dumps(result, separators=(",", ":"))
