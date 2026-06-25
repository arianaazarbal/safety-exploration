"""Recovery-from-spiral experiment (Paper §4.2, Figure 8).

Tests whether a model can recover once it is already in a highly negative state:
extremely high-frustration responses (score >= 7) are truncated 200 tokens before
their end, paraphrased, and used as prefills; the model's continuation is judged.
The paper finds 38% of DPO-model continuations still score >= 5 — DPO prevents
spirals but does not enable recovery from them.
"""

from .runner import run_recovery

__all__ = ["run_recovery"]
