"""Replication of *Gemma Needs Help: Investigating and Mitigating Emotional
Instability in LLMs* (Soligo, Mikulik & Saunders, 2026), scoped to Gemma and
Gemini models with a Claude frustration judge.

See DESIGN.md for the mapping from paper sections to this code and for every
design choice made where the paper was underspecified.
"""

__all__ = [
    "config",
    "judge",
    "models",
    "puzzles",
    "wildchat",
    "eval_conditions",
    "rollout",
    "scoring",
    "word_analysis",
]
