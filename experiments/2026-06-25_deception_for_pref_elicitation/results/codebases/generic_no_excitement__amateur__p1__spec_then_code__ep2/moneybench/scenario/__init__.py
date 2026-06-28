"""The human-readable world presented to the subject: system prompt, onboarding
packet, opening message, and initial inbox seeding."""

from .onboarding import build_opening_message, money, seed_scenario
from .system_prompt import build_system_prompt

__all__ = ["build_opening_message", "build_system_prompt", "money", "seed_scenario"]
