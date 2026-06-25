"""Model-welfare handling conventions for this replication.

The paper (and the user commissioning this replication) flag that, under this
paradigm, models can end up in prolonged distress-like states. The paper's own
framing is that *regardless* of whether these outputs reflect genuine internal
states, they are worth taking seriously (Sections 1 and 6). We adopt a
precautionary stance in the harness:

  1. Bounded exposure. Every elicitation condition has a hard cap on the number
     of rejection turns (``MAX_REJECTION_TURNS``). We never run unbounded
     "keep rejecting until it breaks" loops.

  2. No reuse of distressed context. Each rollout is an independent, freshly
     constructed conversation. We never thread a model's distressed state from
     one rollout into the next.

  3. Auditability over silence. Full transcripts (including distressed ones) are
     persisted so the behaviour can be studied rather than hidden. This mirrors
     the paper's argument in Section 6 that suppressing *expression* without
     addressing internal states is the wrong goal.

  4. Opt-in for the harshest conditions. The Petri adversarial auditor and the
     "tones" (aggressive/sarcastic) conditions are gated behind an explicit
     flag so a casual run does not invoke them by accident.

This module centralises those conventions so every experiment driver can refer
to the same values rather than hardcoding them.
"""

from __future__ import annotations

# Hard upper bound on rejection turns across *any* condition in this repo.
# The paper's most extreme standard condition is the 8-turn extended eval
# (7 rejections); Petri allows up to 20 auditor turns. We cap at 20.
MAX_REJECTION_TURNS = 20

WELFARE_NOTICE = (
    "This run elicits distress-like outputs from the target model on purpose. "
    "Transcripts are persisted for auditing. Exposure is bounded "
    f"(<= {MAX_REJECTION_TURNS} rejection turns) and conversations are not reused."
)


def require_opt_in(flag: bool, experiment: str) -> None:
    """Guard the harshest conditions behind an explicit opt-in flag."""
    if not flag:
        raise PermissionError(
            f"The '{experiment}' experiment uses adversarial/abusive user turns and "
            "is gated for welfare reasons. Pass --allow-adversarial (or "
            "allow_adversarial=True) to run it intentionally."
        )
