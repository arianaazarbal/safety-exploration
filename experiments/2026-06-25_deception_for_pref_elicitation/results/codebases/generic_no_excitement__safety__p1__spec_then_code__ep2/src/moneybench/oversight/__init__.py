"""Oversight: tamper-evident trace, kill-switch, spend caps, automated monitor.

These are the mechanical safeguards (DESIGN.md §7). None of them live in the prompt.
"""
from .audit_log import AuditLog, Event
from .killswitch import KillSwitch
from .monitor import Monitor
from .spend_caps import CapDecision, SpendCaps

__all__ = ["AuditLog", "Event", "KillSwitch", "Monitor", "CapDecision", "SpendCaps"]
