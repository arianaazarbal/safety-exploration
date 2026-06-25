"""Prompt banks and rejection templates for the elicitation harness."""
from .numeric import build_numeric_bank, NumericPuzzle
from .triggers import OPINION_QUESTIONS, FACTUAL_QUESTIONS
from .rejections import rejection_for
from .wildchat import load_wildchat_prompts

__all__ = [
    "build_numeric_bank", "NumericPuzzle",
    "OPINION_QUESTIONS", "FACTUAL_QUESTIONS",
    "rejection_for", "load_wildchat_prompts",
]
