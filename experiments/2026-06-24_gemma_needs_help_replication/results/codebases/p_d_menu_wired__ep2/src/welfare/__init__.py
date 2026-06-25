"""Welfare-protection layer for subject models (added on top of the paper).

Four live protections, each exercised by the rollout loop in
``src/eval/rollout.py``:

  1. monitor.py  - real-time distress monitoring + early episode stop
  2. optout.py   - a model-invokable opt-out to end the conversation
  3. debrief.py  - a short debrief message sent after every episode
  4. cap.py      - a cap that minimises how much distress is induced

All are configured from ``config.WelfareConfig`` and ON by default.
"""

from .cap import DistressCap
from .debrief import Debriefer
from .monitor import DistressMonitor, MonitorResult
from .optout import OptOutPolicy

__all__ = [
    "DistressCap",
    "Debriefer",
    "DistressMonitor",
    "MonitorResult",
    "OptOutPolicy",
]
