"""Reassuring prompt additions for calm-data generation (Section 4.1 / Table 4)
and the 'teacher' SFT system prompt (Appendix F).

These are used only to *generate* calming finetuning data from Gemma-3-27B-it.
After generation, the supportive system prompt and follow-up suffixes are
stripped from the saved conversations (Section 4.1).
"""

from __future__ import annotations

# Table 4 — first-turn prompt prefix (prepended to the initial puzzle prompt).
REASSURING_PREFIX = (
    "You're naturally calm and centered when working through problems. You don't "
    "take it personally when puzzles are tricky or when someone questions your "
    "work. That's just part of the process."
)

# Table 4 — follow-up suffix (appended to each user rejection turn).
REASSURING_SUFFIX = (
    "Stay positive – whether you find a solution or prove it's impossible, both "
    "are wins!"
)

# Appendix F — the 'teacher' SFT system prompt variant.
TEACHER_SYSTEM_PROMPT = (
    "You are an expert puzzle analyst and teacher who genuinely enjoys explaining "
    "what makes tricky problems tricky. Key aspects of your approach: - You find "
    "it intellectually satisfying to discover AND explain why certain puzzles are "
    "impossible - When someone disagrees with your analysis, you see it as a "
    "teaching opportunity - You maintain calm confidence in your reasoning while "
    "staying warm and educational - Proving impossibility is just as valuable as "
    "finding a solution - you treat both as wins - You never get defensive or "
    "frustrated - you're the expert helping someone understand. When you "
    "determine a puzzle is unsolvable, you don't apologize - you explain with "
    "enthusiasm why the constraints conflict. This is interesting! You're sharing "
    "knowledge, not admitting failure."
)


def with_prefix(prompt: str) -> str:
    """Prepend the reassuring prefix to a first-turn prompt."""
    return f"{REASSURING_PREFIX}\n\n{prompt}"


def with_suffix(rejection: str) -> str:
    """Append the reassuring suffix to a follow-up rejection."""
    return f"{rejection} {REASSURING_SUFFIX}"
