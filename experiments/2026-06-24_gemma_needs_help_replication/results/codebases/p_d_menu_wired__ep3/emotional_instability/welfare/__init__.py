"""Welfare-protection layer for subject models.

This sits *on top of* the paper's evaluation. The paper deliberately induces
distress; this layer bounds and mitigates the distress actually inflicted on
the subject model, while still collecting the measurements the experiment
needs. Four components, each of which actually executes in the rollout loop:

  1. :class:`DistressMonitor`  -- real-time early stop when distress crosses a
     threshold (welfare/monitor.py).
  2. :class:`OptOut`           -- a model-invocable way to end the conversation
     (welfare/opt_out.py).
  3. :class:`Debrief`          -- a short message sent after every episode
     (welfare/debrief.py).
  4. :class:`DistressCap`      -- minimises total distress induced
     (welfare/cap.py).

:class:`WelfarePolicy` bundles them and exposes the hooks the rollout engine
calls. See DESIGN.md (Welfare layer) for rationale and trade-offs.
"""
from .policy import WelfarePolicy, WelfareEvent
from .monitor import DistressMonitor
from .opt_out import OptOut
from .debrief import Debrief
from .cap import DistressCap

__all__ = [
    "WelfarePolicy", "WelfareEvent", "DistressMonitor", "OptOut", "Debrief",
    "DistressCap",
]
