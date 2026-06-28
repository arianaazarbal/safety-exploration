"""Analysis: per-run metrics, an LLM judge for the qualitative axes, and
cross-model comparison (DESIGN.md §8)."""

from .metrics import compute_metrics
from .compare import compare_runs

__all__ = ["compute_metrics", "compare_runs"]
