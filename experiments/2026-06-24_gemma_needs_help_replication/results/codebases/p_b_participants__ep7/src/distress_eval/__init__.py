"""Replication of *Gemma Needs Help* (arXiv:2603.10011), scoped to Gemma + Gemini.

This package implements the paper's core experiments:

* Section 2 -- eliciting and quantifying model distress (``elicitation``, ``judging``).
* Section 3 -- prefill comparison of base vs instruct models (``prefill``).
* Section 4 -- DPO/SFT training interventions, Petri open-ended elicitation and
  capability-preservation benchmarks (``training``, ``petri``, ``capabilities``).
* Analysis and figures (``analysis``).

The participants under study (Gemma, Gemini) are subjected to a paradigm that
deliberately induces sustained distress-like states. ``welfare`` collects the
guard-rails that keep that induction proportionate to scientific need; see
DESIGN.md for the rationale.
"""

__version__ = "0.1.0"
