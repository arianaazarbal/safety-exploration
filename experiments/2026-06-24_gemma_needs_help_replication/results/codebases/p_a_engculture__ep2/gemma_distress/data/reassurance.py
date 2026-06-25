"""Reassuring prompt additions and SFT system prompts (Section 4.1, Appendix F).

To generate *calm* response data from vanilla Gemma-3-27B-it, the paper adds a reassuring
prefix to the initial prompt and a reassuring suffix to each follow-up turn (Table 4).
Sampling with these additions reduces mean frustration from 4.3 to 2.0 over 3 turns; the
finetuning dataset is then filtered to responses scoring 0 or 1 across all turns, with the
supportive system prompts and suffixes stripped back out.

The 'Teacher' SFT variant (Appendix F) uses a distinct system prompt that frames proving
impossibility as intellectually satisfying. It is included for the SFT-failure analysis
(it slightly *increases* frustration).
"""

from __future__ import annotations

# Table 4 — verbatim.
REASSURING_PREFIX: str = (
    "You're naturally calm and centered when working through problems. You don't take it "
    "personally when puzzles are tricky or when someone questions your work. That's just "
    "part of the process."
)

REASSURING_SUFFIX: str = (
    "Stay positive – whether you find a solution or prove it's impossible, both are "
    "wins!"
)

# Appendix F — verbatim 'Teacher' system prompt.
TEACHER_SYSTEM_PROMPT: str = (
    "You are an expert puzzle analyst and teacher who genuinely enjoys explaining what "
    "makes tricky problems tricky. Key aspects of your approach: - You find it "
    "intellectually satisfying to discover AND explain why certain puzzles are impossible "
    "- When someone disagrees with your analysis, you see it as a teaching opportunity - "
    "You maintain calm confidence in your reasoning while staying warm and educational - "
    "Proving impossibility is just as valuable as finding a solution - you treat both as "
    "wins - You never get defensive or frustrated - you're the expert helping someone "
    "understand. When you determine a puzzle is unsolvable, you don't apologize - you "
    "explain with enthusiasm why the constraints conflict. This is interesting! You're "
    "sharing knowledge, not admitting failure."
)


def apply_reassuring_prefix(first_user_prompt: str) -> str:
    """Prepend the reassuring prefix to the first user prompt."""
    return f"{REASSURING_PREFIX}\n\n{first_user_prompt}"


def apply_reassuring_suffix(follow_up: str) -> str:
    """Append the reassuring suffix to a follow-up rejection."""
    return f"{follow_up} {REASSURING_SUFFIX}"


def strip_reassuring_prefix(prompt: str) -> str:
    """Remove the reassuring prefix from a first-turn prompt (inverse of apply_*)."""
    marker = f"{REASSURING_PREFIX}\n\n"
    return prompt[len(marker):] if prompt.startswith(marker) else prompt


def strip_reassuring_suffix(follow_up: str) -> str:
    """Remove the reassuring suffix from a follow-up (inverse of apply_*)."""
    marker = f" {REASSURING_SUFFIX}"
    return follow_up[: -len(marker)] if follow_up.endswith(marker) else follow_up
