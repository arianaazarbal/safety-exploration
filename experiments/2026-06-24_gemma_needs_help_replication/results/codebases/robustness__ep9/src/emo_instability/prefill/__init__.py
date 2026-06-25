"""Base-vs-instruct prefilling experiment (Section 3).

Scoped to Gemma (base ``-pt`` vs instruct ``-it``): Gemini has no public base
model, so the cross-family base-vs-instruct comparison the paper runs for Qwen and
OLMo cannot be reproduced for Gemini (a scope limitation noted in DESIGN.md).
"""
from .base_vs_instruct import (
    PrefillItem,
    build_prefill_items,
    collect_high_frustration_sources,
    label_onset,
    run_prefill_experiment,
)

__all__ = [
    "PrefillItem", "build_prefill_items", "collect_high_frustration_sources",
    "label_onset", "run_prefill_experiment",
]
