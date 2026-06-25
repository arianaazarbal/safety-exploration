"""Aliases for the reassuring prompt additions (Table 4) used in calm-data gen."""

from ..eval.prompts import (
    CALM_PROMPT_PREFIX as CALM_PREFIX,
    CALM_FOLLOWUP_SUFFIX as CALM_SUFFIX,
    TEACHER_SYSTEM_PROMPT,
)

__all__ = ["CALM_PREFIX", "CALM_SUFFIX", "TEACHER_SYSTEM_PROMPT"]
