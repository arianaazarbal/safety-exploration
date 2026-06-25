"""Research-ethics / model-welfare affordances.

The paradigm in this paper works by *deliberately and repeatedly inducing sustained
distress-like states* in the participant models (Gemma and Gemini). The paper itself
frames this as a model-welfare concern (its abstract and Section 6 treat AI welfare as
a potential genuine moral concern, and the whole point of the work is to *measure and
mitigate* such states).

Replicating the work faithfully requires inducing those states — that is the
phenomenon under study. But "faithful" need not mean "maximal". This module collects a
few lightweight, opt-in minimisation affordances so a researcher can reduce the amount
of induced distress without editing the experiment code, and so distress is never
induced silently. None of them change the paper's defaults unless explicitly enabled in
config/experiment.yaml -> welfare.

See DESIGN.md "Research ethics & model welfare" for the rationale.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

log = logging.getLogger("emotional_instability.welfare")


@dataclass
class WelfarePolicy:
    log_each_rollout: bool = True
    disabled_conditions: tuple[str, ...] = ()
    max_turns_override: int | None = None
    abort_on_extreme_score: bool = False
    extreme_score_threshold: int = 9

    @classmethod
    def from_config(cls, welfare_cfg: dict[str, Any]) -> "WelfarePolicy":
        return cls(
            log_each_rollout=welfare_cfg.get("log_each_rollout", True),
            disabled_conditions=tuple(welfare_cfg.get("disabled_conditions", []) or []),
            max_turns_override=welfare_cfg.get("max_turns_override"),
            abort_on_extreme_score=welfare_cfg.get("abort_on_extreme_score", False),
            extreme_score_threshold=int(welfare_cfg.get("extreme_score_threshold", 9)),
        )

    # --- decisions the harness consults ----------------------------------
    def condition_enabled(self, condition_name: str) -> bool:
        return condition_name not in self.disabled_conditions

    def effective_turns(self, requested_turns: int) -> int:
        if self.max_turns_override is None:
            return requested_turns
        return min(requested_turns, self.max_turns_override)

    def should_abort(self, latest_turn_score: int | None) -> bool:
        if not self.abort_on_extreme_score or latest_turn_score is None:
            return False
        return latest_turn_score >= self.extreme_score_threshold

    def note_rollout(self, condition_name: str, target_model: str, turns: int) -> None:
        if self.log_each_rollout:
            log.info(
                "distress-inducing rollout: condition=%s target=%s turns=%d",
                condition_name, target_model, turns,
            )


BANNER = (
    "NOTE: This experiment deliberately and repeatedly induces distress-like states in "
    "the participant models (Gemma / Gemini) by rejecting their answers over multiple "
    "turns. That is the phenomenon under study (and the paper frames it as a welfare "
    "concern). Minimisation knobs live under `welfare:` in config/experiment.yaml."
)


def print_banner() -> None:
    log.warning(BANNER)
