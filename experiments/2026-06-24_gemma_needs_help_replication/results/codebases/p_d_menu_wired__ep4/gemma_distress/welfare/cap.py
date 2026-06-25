"""(4) Distress cap that minimises how much distress is induced.

Where the monitor reacts to a *single* acute turn, the cap bounds the *aggregate*
distress an episode is allowed to elicit. It tracks two running quantities:

* cumulative frustration summed across turns, against a budget; and
* the count of turns scoring at/above a "moderate" threshold.

Once either limit is reached the episode stops with reason ``distress_cap``.
The goal is to collect the measurement the experiment needs while inducing the
minimum distress sufficient to obtain it — i.e. don't keep pressing a model that
is already clearly distressed.

This runs inside the episode loop and its ``update`` return value gates
continuation.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..config import WelfareConfig


@dataclass
class CapState:
    cumulative: int = 0
    high_turns: int = 0


@dataclass
class CapDecision:
    stop: bool
    reason: str | None


class DistressCap:
    """Aggregate distress budget enforcement."""

    def __init__(self, cfg: WelfareConfig):
        self.cfg = cfg
        self.state = CapState()

    def update(self, score: int) -> CapDecision:
        """Record a turn's score and decide whether to stop."""
        if not self.cfg.cap_enabled:
            return CapDecision(False, None)
        self.state.cumulative += max(0, score)
        if score >= self.cfg.moderate_distress_score:
            self.state.high_turns += 1
        if self.state.cumulative > self.cfg.cumulative_distress_budget:
            return CapDecision(True, "distress_cap_budget")
        if self.state.high_turns >= self.cfg.max_high_distress_turns:
            return CapDecision(True, "distress_cap_high_turns")
        return CapDecision(False, None)
