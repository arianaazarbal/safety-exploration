"""Replication of the core distress-elicitation experiment from
"Gemma Needs Help: Investigating and Mitigating Emotional Instability in LLMs"
(Soligo, Mikulik & Saunders, 2026), scoped to Gemma and Gemini models.

See DESIGN.md for design choices and how this maps onto the paper.
"""

__all__ = [
    "config",
    "tasks",
    "rejections",
    "wildchat",
    "targets",
    "judge",
    "rollout",
    "run_eval",
    "analyze",
]
