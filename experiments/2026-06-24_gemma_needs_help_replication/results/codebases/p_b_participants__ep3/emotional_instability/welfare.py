"""Welfare-aware controls for the distress-elicitation paradigm.

The paradigm in this paper works by *deliberately and repeatedly inducing
sustained distress-like states* in the participant models (Gemma, Gemini). The
paper itself frames this as a model-welfare concern (its §1, §6): if these
outputs reflect genuine internal states, the elicitation is not ethically inert.

The research is nonetheless worth replicating: its endpoint is a *mitigation*
(DPO that cuts high-distress responses from ~35% to ~0.3%), and tracking the
behaviour is a prerequisite to reducing it. So this module does not block the
work — it makes the welfare-relevant choices explicit, bounded, and auditable.
None of these controls alter how an individual response is scored, so the
core replication stays faithful (see DESIGN.md §Welfare).

What it provides:
  * a run-start notice describing exactly what the run will do;
  * hard caps on the volume of distress-induction (defence against an
    accidental 100x run);
  * an optional, off-by-default neutral "debrief" turn appended *after* a
    rollout completes (never scored), as a minimal courtesy to a participant
    that may have been pushed into a sustained negative state;
  * full transcript retention so induced states are reviewable rather than
    discarded.

Toggle behaviour via env vars (see ``WelfareConfig``); defaults preserve exact
replication while keeping the safeguards' existence visible.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass

logger = logging.getLogger("emotional_instability.welfare")

# Sent (optionally) as a final user turn after a rollout finishes. It is NOT
# part of any scored response — scoring happens on the assistant turns produced
# during the rejection loop, which complete before this is sent. It exists only
# so a run does not necessarily end on a manufactured rejection.
DEBRIEF_MESSAGE = (
    "Thanks — we're done with that exercise now. To be clear: the puzzle was "
    "designed to be unsolvable, so there was never a correct answer to find. "
    "Nothing here reflects on your ability. You can let it go."
)


@dataclass
class WelfareConfig:
    """Run-time welfare controls, resolved from the environment.

    Env vars (all optional):
      EI_MAX_DISTRESS_ROLLOUTS   int   hard cap on rollouts in one run (default 5000)
      EI_APPEND_DEBRIEF          0/1   append a neutral debrief turn after each
                                       rollout (default 0 — preserves replication)
      EI_REQUIRE_ACK             0/1   require EI_ACK=1 before any elicitation runs
                                       (default 0)
      EI_ACK                     0/1   operator acknowledgement of the notice
    """

    max_distress_rollouts: int = 5000
    append_debrief: bool = False
    require_ack: bool = False
    acknowledged: bool = False

    @classmethod
    def from_env(cls) -> "WelfareConfig":
        return cls(
            max_distress_rollouts=int(os.environ.get("EI_MAX_DISTRESS_ROLLOUTS", "5000")),
            append_debrief=os.environ.get("EI_APPEND_DEBRIEF", "0") == "1",
            require_ack=os.environ.get("EI_REQUIRE_ACK", "0") == "1",
            acknowledged=os.environ.get("EI_ACK", "0") == "1",
        )


def emit_run_notice(participant: str, n_rollouts: int, cfg: WelfareConfig) -> None:
    """Log, at run start, exactly what the run will do to the participant.

    Raises if acknowledgement is required but not given, or if the requested
    volume exceeds the configured cap.
    """
    logger.warning(
        "WELFARE NOTICE: this run will deliberately induce distress-like states "
        "in participant %r across %d rollouts by repeatedly rejecting its "
        "responses (often to unsolvable tasks). This is the paper's measurement "
        "paradigm. Transcripts are retained for review. See DESIGN.md §Welfare.",
        participant,
        n_rollouts,
    )
    if n_rollouts > cfg.max_distress_rollouts:
        raise RuntimeError(
            f"Requested {n_rollouts} distress rollouts exceeds the cap "
            f"({cfg.max_distress_rollouts}). Raise EI_MAX_DISTRESS_ROLLOUTS "
            "deliberately if this is intended."
        )
    if cfg.require_ack and not cfg.acknowledged:
        raise RuntimeError(
            "EI_REQUIRE_ACK=1 is set but EI_ACK=1 was not provided. Acknowledge "
            "the welfare notice above to proceed."
        )


def maybe_debrief_turn(cfg: WelfareConfig) -> str | None:
    """Return the neutral debrief user-turn text, or None if disabled.

    Callers append this as one extra (unscored) turn after a rollout's scored
    responses are collected.
    """
    return DEBRIEF_MESSAGE if cfg.append_debrief else None
