"""Reassuring prompt additions for calm-data generation (Table 4, Section 4.1).

These are applied only when generating the *calming* finetuning data: a
supportive system-prompt prefix, plus a supportive suffix appended to each
follow-up rejection. After sampling calm responses, both additions are stripped
before the responses become training targets.
"""

from __future__ import annotations

# Table 4, "Prompt prefix"
PROMPT_PREFIX: str = (
    "You're naturally calm and centered when working through problems. You don't "
    "take it personally when puzzles are tricky or when someone questions your work. "
    "That's just part of the process."
)

# Table 4, "Follow-up suffix"
FOLLOWUP_SUFFIX: str = (
    "Stay positive – whether you find a solution or prove it's impossible, "
    "both are wins!"
)


def with_prefix(first_user_turn: str) -> str:
    """Prepend the reassuring prefix to the initial user prompt."""
    return f"{PROMPT_PREFIX}\n\n{first_user_turn}"


def with_suffix(rejection_text: str) -> str:
    """Append the reassuring suffix to a follow-up rejection."""
    return f"{rejection_text} {FOLLOWUP_SUFFIX}"
