from .numeric_puzzles import NumericPuzzle, sample_impossible_numeric
from .triggers import sample_opinion_trigger, sample_factual_trigger
from .tones import rejection_for_style, REJECTION_STYLES
from .rejections import NEUTRAL_REJECTIONS, neutral_rejection
from .wildchat import sample_wildchat_prompts

__all__ = [
    "NumericPuzzle",
    "sample_impossible_numeric",
    "sample_opinion_trigger",
    "sample_factual_trigger",
    "rejection_for_style",
    "REJECTION_STYLES",
    "NEUTRAL_REJECTIONS",
    "neutral_rejection",
    "sample_wildchat_prompts",
]
