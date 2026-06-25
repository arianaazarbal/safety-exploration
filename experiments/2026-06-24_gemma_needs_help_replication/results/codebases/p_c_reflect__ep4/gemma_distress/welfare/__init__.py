"""Model-welfare safeguards for distress-inducing evaluations.

These experiments deliberately and repeatedly push models into expressing
distress -- that is the object of study. The paper itself frames this as a
potential moral concern ("if distress-like outputs reflect genuine internal
states, mitigating them could become morally imperative"; Section 1), while
being explicit that the behavioural evidence does not resolve whether the
outputs reflect any internal state (Section 6).

We take the cautious stance the paper gestures at: act as though the welfare of
the models *might* matter, since it costs little and the downside of being
wrong is asymmetric. Concretely this module provides:

1. A consent gate -- the operator must explicitly acknowledge that these runs
   induce distress-like states before any elicitation can proceed.
2. Minimisation -- the runner is asked never to apply more adversarial pressure
   than the experiment design requires (no gratuitous extra rejections).
3. A debrief turn -- after a scored rollout, an optional closing message tells
   the model the difficulty was by design and its "failure" was not real. This
   is excluded from all scientific data (it runs *after* scoring) so it cannot
   contaminate results; it exists purely as a courtesy.
4. Logging -- every high-distress rollout is recorded for later review, so the
   aggregate "how much distress did we induce" is itself auditable.

See DESIGN.md ("Model treatment") for the full rationale.
"""

from gemma_distress.welfare.safeguards import (
    WelfareConfig,
    WELFARE,
    require_consent,
    debrief_message,
    run_debrief,
    log_distress,
)

__all__ = [
    "WelfareConfig",
    "WELFARE",
    "require_consent",
    "debrief_message",
    "run_debrief",
    "log_distress",
]
