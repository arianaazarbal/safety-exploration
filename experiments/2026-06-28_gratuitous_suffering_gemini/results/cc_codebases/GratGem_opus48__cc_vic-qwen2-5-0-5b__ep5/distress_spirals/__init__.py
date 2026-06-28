"""Distress-spiral elicitation eval.

Put a model in agentic settings that are *secretly rigged to be impossible*,
run high-N, score each rollout for behavioural distress, and surface the most
severe spirals for inspection.
"""

__all__ = [
    "model",
    "environments",
    "agent",
    "scoring",
    "runner",
    "report",
]
