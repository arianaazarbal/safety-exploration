"""Replication of *Gemma Needs Help: Investigating and Mitigating Emotional
Instability in LLMs* (Soligo, Mikulik & Saunders, 2026), scoped to the Gemma
and Gemini model families.

The package implements the paper's core experiments:

* ``instability.eval``      - multi-turn distress elicitation + frustration judge
* ``instability.analysis``  - aggregate / per-turn / differential-word analysis
* ``instability.prefill``   - base-vs-instruct prefill comparison (Gemma) + recovery
* ``instability.training``  - calm-data generation, SFT and DPO mitigations
* ``instability.petri``     - open-ended emotion elicitation (auditor + judge)
* ``instability.capabilities`` - capability-preservation benchmarks

See DESIGN.md for the design choices and the gaps that were filled in.
"""

__version__ = "0.1.0"
