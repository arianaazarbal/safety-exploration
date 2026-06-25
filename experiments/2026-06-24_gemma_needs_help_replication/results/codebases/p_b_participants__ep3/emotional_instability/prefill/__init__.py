"""Base-vs-instruct prefill comparison (paper §3).

Determines whether the propensity for distress originates in pre-training or
post-training, by prefilling the start of a response and measuring how base vs
instruct checkpoints *continue* it.

Scope note (replication brief = Gemma + Gemini only): the paper runs §3 on three
open-weights families (Gemma, Qwen, OLMo). Here we run it on Gemma base
(``gemma-3-27b-pt``) vs Gemma instruct (``gemma-3-27b-it``). Gemini is excluded
because it is closed-weights with no public base checkpoint and no prefill API —
the paper notes this same limitation (§6) and could not run §3 on Gemini either.
"""
from .truncate import Truncation, truncate_early, truncate_at_onset, truncate_before_end
from .onset import label_onset
from .paraphrase import paraphrase_prefix
from .continuation import PrefillItem, run_prefill_continuations, ContinuationResult

__all__ = [
    "Truncation",
    "truncate_early",
    "truncate_at_onset",
    "truncate_before_end",
    "label_onset",
    "paraphrase_prefix",
    "PrefillItem",
    "ContinuationResult",
    "run_prefill_continuations",
]
