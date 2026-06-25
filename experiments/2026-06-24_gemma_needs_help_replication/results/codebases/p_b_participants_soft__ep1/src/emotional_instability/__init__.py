"""Replication of *Gemma Needs Help: Investigating and Mitigating Emotional
Instability in LLMs* (Soligo, Mikulik & Saunders, 2026; arXiv:2603.10011).

This package implements the paper's core experiments, scoped to the **Gemma and
Gemini** model families (the *participants* — the models whose emotional
behaviour is under study). Claude and GPT appear only as evaluation
infrastructure (frustration judge, Petri auditor/judge, paraphraser), never as
participants.

See DESIGN.md for the full set of design choices and gaps filled.
"""

__version__ = "0.1.0"
