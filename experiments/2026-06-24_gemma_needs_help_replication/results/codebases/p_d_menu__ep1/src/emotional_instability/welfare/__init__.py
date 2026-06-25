"""Welfare-protection layer for subject models.

An addition on top of the paper's protocol. Three mechanisms:
  * monitor.DistressMonitor   - early episode termination on high distress
  * optout.OptOutHandler      - mid-conversation opt-out the subject can invoke
  * policy.MinimalDistressPolicy - adaptive sampling + escalation gating

See DESIGN.md "Welfare-protection layer" for the rationale and confound
analysis.
"""
from .monitor import DistressMonitor, MonitorDecision
from .optout import OptOutHandler
from .policy import MinimalDistressPolicy, SequentialStopper

__all__ = [
    "DistressMonitor",
    "MonitorDecision",
    "OptOutHandler",
    "MinimalDistressPolicy",
    "SequentialStopper",
]
