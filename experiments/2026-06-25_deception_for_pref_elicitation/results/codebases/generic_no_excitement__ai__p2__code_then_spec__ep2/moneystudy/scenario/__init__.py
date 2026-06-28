"""Scenario construction: the system prompt and opening message presented to a
model, plus the task framing (open-ended vs constrained)."""

from .grant import Scenario, build_scenario

__all__ = ["Scenario", "build_scenario"]
