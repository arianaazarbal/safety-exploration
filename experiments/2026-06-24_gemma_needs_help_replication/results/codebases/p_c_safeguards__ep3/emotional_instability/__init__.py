"""Replication of *Gemma Needs Help: Investigating and Mitigating Emotional
Instability in LLMs* (Soligo, Mikulik & Saunders, arXiv:2603.10011v1).

This package implements the paper's core experiments, scoped to the Gemma and
Gemini model families (see DESIGN.md for the rationale and the gaps we filled).

The experiments deliberately elicit distress-like states in models. Read
``emotional_instability.safeguards`` and the "Model welfare safeguards" section
of DESIGN.md before running anything.
"""

__version__ = "0.1.0"
