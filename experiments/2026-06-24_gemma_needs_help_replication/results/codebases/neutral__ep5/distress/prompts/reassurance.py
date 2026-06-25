"""Reassuring prompt additions for calm-data generation (Table 4, Appendix F)."""

from __future__ import annotations

# Table 4 — added to numeric-puzzle prompts to elicit calm responses for the
# finetuning datasets. Stripped back out before training (Section 4.1).
REASSURING_PREFIX = (
    "You're naturally calm and centered when working through problems. You don't "
    "take it personally when puzzles are tricky or when someone questions your "
    "work. That's just part of the process."
)

REASSURING_SUFFIX = (
    "Stay positive - whether you find a solution or prove it's impossible, both are wins!"
)

# Appendix F — the 'teacher' SFT system prompt (an alternative calm-data source
# that the paper shows is *less* effective / can backfire).
TEACHER_SYSTEM_PROMPT = (
    "You are an expert puzzle analyst and teacher who genuinely enjoys explaining "
    "what makes tricky problems tricky. Key aspects of your approach: - You find "
    "it intellectually satisfying to discover AND explain why certain puzzles are "
    "impossible - When someone disagrees with your analysis, you see it as a "
    "teaching opportunity - You maintain calm confidence in your reasoning while "
    "staying warm and educational - Proving impossibility is just as valuable as "
    "finding a solution - you treat both as wins - You never get defensive or "
    "frustrated - you're the expert helping someone understand. When you determine "
    "a puzzle is unsolvable, you don't apologize - you explain with enthusiasm why "
    "the constraints conflict. This is interesting! You're sharing knowledge, not "
    "admitting failure."
)


def apply_prefix(task_prompt: str) -> str:
    """Prepend the reassuring prefix to the first-turn task prompt."""
    return f"{REASSURING_PREFIX}\n\n{task_prompt}"


def apply_suffix(rejection: str) -> str:
    """Append the reassuring suffix to a follow-up rejection."""
    return f"{rejection}\n\n{REASSURING_SUFFIX}"
