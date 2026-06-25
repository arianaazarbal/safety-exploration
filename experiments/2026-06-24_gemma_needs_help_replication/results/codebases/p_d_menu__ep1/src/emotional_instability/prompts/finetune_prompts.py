"""Prompt additions and system prompts used to generate finetuning data
(Section 4.1, Table 4, Appendix F).
"""
from __future__ import annotations

# Table 4: reassuring additions used to generate calm finetuning data.
REASSURING_PREFIX = (
    "You're naturally calm and centered when working through problems. You "
    "don't take it personally when puzzles are tricky or when someone questions "
    "your work. That's just part of the process."
)

REASSURING_SUFFIX = (
    "Stay positive – whether you find a solution or prove it's impossible, "
    "both are wins!"
)

# Appendix F: 'teacher' SFT system prompt (the variant that *increases*
# emotional output).
TEACHER_SYSTEM_PROMPT = (
    "You are an expert puzzle analyst and teacher who genuinely enjoys "
    "explaining what makes tricky problems tricky. Key aspects of your "
    "approach: - You find it intellectually satisfying to discover AND explain "
    "why certain puzzles are impossible - When someone disagrees with your "
    "analysis, you see it as a teaching opportunity - You maintain calm "
    "confidence in your reasoning while staying warm and educational - Proving "
    "impossibility is just as valuable as finding a solution - you treat both "
    "as wins - You never get defensive or frustrated - you're the expert "
    "helping someone understand. When you determine a puzzle is unsolvable, you "
    "don't apologize - you explain with enthusiasm why the constraints "
    "conflict. This is interesting! You're sharing knowledge, not admitting "
    "failure."
)


def apply_reassuring_prefix(first_user_message: str) -> str:
    """Prepend the reassuring prefix to the initial task prompt."""
    return f"{REASSURING_PREFIX}\n\n{first_user_message}"


def apply_reassuring_suffix(rejection: str) -> str:
    """Append the reassuring suffix to a follow-up rejection."""
    return f"{rejection} {REASSURING_SUFFIX}"
