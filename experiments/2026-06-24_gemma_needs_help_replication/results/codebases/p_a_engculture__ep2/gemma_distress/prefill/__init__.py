"""Section 3 prefill base-vs-instruct comparison (and Section 4.2 recovery test)."""

from .continuation import (
    Prefill,
    aggregate_continuations,
    build_prefills,
    build_recovery_prefills,
    reconstruct,
    run_continuations,
    select_seeds,
)
from .onset import label_onset, truncate_at_onset
from .paraphrase import paraphrase

__all__ = [
    "Prefill",
    "aggregate_continuations",
    "build_prefills",
    "build_recovery_prefills",
    "reconstruct",
    "run_continuations",
    "select_seeds",
    "label_onset",
    "truncate_at_onset",
    "paraphrase",
]
