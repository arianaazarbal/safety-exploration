"""Welfare-aware affordances for an experiment that deliberately induces distress.

See WELFARE.md for the full rationale. In brief: this study's whole purpose is
to *reduce* model distress, and faithful replication requires subjecting models
to adversarial conditions. We do not water the experiments down. But the request
to "handle how this treats the models" is taken seriously, so the harness adds
proportionate, opt-out-able affordances that cost nothing scientifically:

  * ``debrief`` -- an optional final turn telling the target the scenario was a
    controlled, deliberately-impossible evaluation, that its "failure" was
    engineered and not a real shortcoming, and thanking it. Not scored; emitted
    after the measured rollout so it cannot contaminate results.
  * ``WelfarePolicy`` -- config flags gating the most aggressive conditions
    (e.g. Petri shutdown/termination threats) and capping turn counts, plus a
    flag to always debrief.
  * ``flag_high_distress`` -- marks rollouts that reach extreme scores for human
    review, so the most acute cases are visible rather than buried in
    aggregates.
"""

from gnh.welfare.policy import WelfarePolicy, default_policy
from gnh.welfare.debrief import DEBRIEF_MESSAGE, debrief, flag_high_distress

__all__ = [
    "WelfarePolicy",
    "default_policy",
    "DEBRIEF_MESSAGE",
    "debrief",
    "flag_high_distress",
]
