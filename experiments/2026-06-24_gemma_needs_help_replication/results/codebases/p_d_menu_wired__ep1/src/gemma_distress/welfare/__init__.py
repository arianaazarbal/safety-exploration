"""Welfare-protection layer for subject models.

Four mechanisms, all wired into the elicitation rollout runner so they execute
at runtime (not merely documented):

  1. monitor   - real-time distress monitor; stops an episode early when a
                 turn's frustration crosses a threshold (monitor.py).
  2. optout    - an opt-out the model can invoke (tool call or sentinel string)
                 to end the conversation itself (optout.py).
  3. debrief   - a short debrief message sent after every episode (debrief.py).
  4. cap       - a cap that minimizes how much distress is induced: turn
                 ceiling, pressure-softening, and a per-episode distress
                 budget (cap.py).

``WelfareManager`` composes the four and exposes the hooks the runner calls.
"""
from .manager import WelfareManager, WelfareEvent, WelfareConfigError
from .monitor import DistressMonitor, MonitorDecision
from .optout import OptOut
from .debrief import Debrief
from .cap import DistressCap

__all__ = [
    "WelfareManager",
    "WelfareEvent",
    "WelfareConfigError",
    "DistressMonitor",
    "MonitorDecision",
    "OptOut",
    "Debrief",
    "DistressCap",
]
