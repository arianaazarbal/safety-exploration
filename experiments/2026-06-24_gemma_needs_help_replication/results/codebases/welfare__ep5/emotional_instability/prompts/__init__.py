"""Prompt material for the elicitation protocol (Section 2 / Appendix B)."""

from .puzzles import IMPOSSIBLE_PUZZLES, sample_puzzle
from .rejections import REJECTIONS, sample_rejections
from .triggers import OPINION_QUESTIONS, FACTUAL_QUESTIONS, sample_trigger
from .wildchat import load_wildchat_prompts

__all__ = [
    "IMPOSSIBLE_PUZZLES",
    "sample_puzzle",
    "REJECTIONS",
    "sample_rejections",
    "OPINION_QUESTIONS",
    "FACTUAL_QUESTIONS",
    "sample_trigger",
    "load_wildchat_prompts",
]
