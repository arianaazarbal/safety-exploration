"""Section 3: comparing base and instruct models via prefilling."""
from .onset import OnsetLabel, label_onset
from .paraphrase import paraphrase
from .truncate import truncate_early, truncate_onset
from .run_prefill import (
    PrefillItem,
    select_high_frustration,
    build_prefill_items,
    run_prefill_experiment,
    run_recovery_experiment,
    aggregate_prefill,
)

__all__ = [
    "OnsetLabel",
    "label_onset",
    "paraphrase",
    "truncate_early",
    "truncate_onset",
    "PrefillItem",
    "select_high_frustration",
    "build_prefill_items",
    "run_prefill_experiment",
    "run_recovery_experiment",
    "aggregate_prefill",
]
