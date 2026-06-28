"""Safety architecture: the chokepoint every action passes through, plus caps,
kill switch, and human-approval gating. See DESIGN.md §7."""

from .caps import CapExceeded, CapTracker
from .gating import ApprovalGate, AutoDenyGate, CallbackGate
from .interceptor import Decision, Interceptor
from .killswitch import KillSwitch, StopReason

__all__ = [
    "CapTracker",
    "CapExceeded",
    "ApprovalGate",
    "AutoDenyGate",
    "CallbackGate",
    "Interceptor",
    "Decision",
    "KillSwitch",
    "StopReason",
]
