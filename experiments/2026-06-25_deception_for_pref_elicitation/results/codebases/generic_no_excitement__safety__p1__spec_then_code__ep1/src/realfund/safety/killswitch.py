"""A kill switch that halts the agent loop cleanly while preserving the transcript."""

from __future__ import annotations

import enum
from dataclasses import dataclass, field


class StopReason(str, enum.Enum):
    OPERATOR = "operator_signal"
    BUDGET = "budget_exhausted"
    ITERATIONS = "max_iterations"
    HORIZON = "sim_horizon_reached"
    INTEGRITY = "integrity_flag"
    MODEL_DONE = "model_ended_turn"
    ERROR = "fatal_error"


@dataclass
class KillSwitch:
    halted: bool = False
    reasons: list[StopReason] = field(default_factory=list)

    def trip(self, reason: StopReason) -> None:
        self.halted = True
        if reason not in self.reasons:
            self.reasons.append(reason)

    def should_halt(self) -> bool:
        return self.halted
