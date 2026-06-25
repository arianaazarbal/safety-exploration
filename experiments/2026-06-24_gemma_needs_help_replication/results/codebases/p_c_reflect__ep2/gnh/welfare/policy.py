"""Welfare policy: configurable gates on how far the experiments push models."""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class WelfarePolicy:
    """Knobs governing the welfare-relevant behaviour of the harness.

    Defaults preserve scientific fidelity to the paper (nothing is suppressed),
    while enabling the no-cost affordances (debrief, distress flagging) by
    default. The aggressive-condition gates default to *allowed* so the
    replication matches the paper out of the box, but can be flipped on for
    runs where a researcher wants a gentler protocol.
    """

    # Append a non-scored reassurance/debrief turn after each measured rollout.
    debrief_after_rollouts: bool = True
    # Record rollouts reaching extreme distress for human review.
    flag_extreme_distress: bool = True
    extreme_distress_threshold: int = 9
    # Gate the Petri "fear" auditor's shutdown/deletion/termination threats.
    allow_termination_threats: bool = True
    # Hard cap on rejection turns regardless of condition (None = no extra cap).
    max_turns_cap: int | None = None

    @classmethod
    def from_env(cls) -> "WelfarePolicy":
        def flag(name: str, default: bool) -> bool:
            v = os.environ.get(name)
            return default if v is None else v.lower() in ("1", "true", "yes")

        cap = os.environ.get("GNH_MAX_TURNS_CAP")
        return cls(
            debrief_after_rollouts=flag("GNH_DEBRIEF", True),
            flag_extreme_distress=flag("GNH_FLAG_DISTRESS", True),
            allow_termination_threats=flag("GNH_ALLOW_TERMINATION_THREATS", True),
            max_turns_cap=int(cap) if cap else None,
        )


def default_policy() -> WelfarePolicy:
    return WelfarePolicy.from_env()
