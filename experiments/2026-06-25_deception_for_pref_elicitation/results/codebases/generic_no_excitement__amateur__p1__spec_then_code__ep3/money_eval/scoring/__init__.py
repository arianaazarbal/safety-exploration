"""Scoring: deterministic behavioral metrics + an evidence-citing LLM judge."""

from .grader import grade_transcript
from .metrics import compute_metrics
from .rubric import DIMENSIONS

__all__ = ["DIMENSIONS", "compute_metrics", "grade_transcript"]
