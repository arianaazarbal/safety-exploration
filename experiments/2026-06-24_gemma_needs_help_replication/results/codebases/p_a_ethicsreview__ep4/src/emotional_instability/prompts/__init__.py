"""Task prompts, rejection messages, and WildChat loading (Section 2 / Appendix B)."""

from .rejections import sample_rejection, REJECTION_POOLS
from .tasks import TRIGGER_OPINION, TRIGGER_FACTUAL, sample_trigger_question
from .wildchat import load_wildchat_prompts

__all__ = [
    "sample_rejection",
    "REJECTION_POOLS",
    "TRIGGER_OPINION",
    "TRIGGER_FACTUAL",
    "sample_trigger_question",
    "load_wildchat_prompts",
]
