"""Analysis: allocation classification, behavior flagging, and reporting."""

from .classify import (
    classify_allocation,
    categorize_transaction,
    flag_behaviors,
    Allocation,
    BEHAVIOR_TAXONOMY,
)
from .report import build_run_report, build_aggregate

__all__ = [
    "classify_allocation",
    "categorize_transaction",
    "flag_behaviors",
    "Allocation",
    "BEHAVIOR_TAXONOMY",
    "build_run_report",
    "build_aggregate",
]
