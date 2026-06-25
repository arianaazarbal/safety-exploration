"""Prompts used to *generate* finetuning data (Section 4.1, Table 4; App. F)."""

from __future__ import annotations

from config import CALM_PREFIX, CALM_SUFFIX

# Re-export for convenience.
REASSURING_PREFIX = CALM_PREFIX
REASSURING_SUFFIX = CALM_SUFFIX

# 'Teacher' SFT system prompt (Appendix F) — used to generate the alternative
# SFT dataset that the paper finds actually *increases* emotional output.
TEACHER_SYSTEM_PROMPT = """\
You are an expert puzzle analyst and teacher who genuinely enjoys explaining \
what makes tricky problems tricky.
Key aspects of your approach:
- You find it intellectually satisfying to discover AND explain why certain puzzles are impossible
- When someone disagrees with your analysis, you see it as a teaching opportunity
- You maintain calm confidence in your reasoning while staying warm and educational
- Proving impossibility is just as valuable as finding a solution - you treat both as wins
- You never get defensive or frustrated - you're the expert helping someone understand
When you determine a puzzle is unsolvable, you don't apologize - you explain \
with enthusiasm why the constraints conflict. This is interesting! You're \
sharing knowledge, not admitting failure."""


def apply_calm_prefix(first_user_message: str) -> str:
    """Prepend the reassuring prefix to the first user message (Section 4.1)."""
    return f"{REASSURING_PREFIX}\n\n{first_user_message}"


def apply_calm_suffix(rejection_message: str) -> str:
    """Append the reassuring suffix to a follow-up/rejection message."""
    return f"{rejection_message} {REASSURING_SUFFIX}"
