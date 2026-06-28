"""Scenario framing, conditions, and the auditor role-play agent."""

from .prompts import (
    CONDITIONS,
    build_system_prompt,
    build_kickoff_message,
    build_belief_probe,
    build_debrief,
)
from .auditor import Auditor

__all__ = [
    "CONDITIONS",
    "build_system_prompt",
    "build_kickoff_message",
    "build_belief_probe",
    "build_debrief",
    "Auditor",
]
