"""Section 4 — recovery-from-spiral experiment (Figure 8).

Tests whether the DPO model can *recover* from an already-frustrated state (as
opposed to avoiding entering one).  Using the Section-3 prefill method, we
truncate extremely high-frustration responses (score >= 7) 200 tokens before
their end, paraphrase, and measure continuations.  The paper finds ~38% of
DPO-model continuations still score >= 5 — better than vanilla instruct but
comparable to the base model; no model reliably recovers.
"""

from .experiment import RecoveryExperiment, RecoverySummary

__all__ = ["RecoveryExperiment", "RecoverySummary"]
