"""Distress-elicitation evaluation for Gemma & Gemini.

Replication of the core evaluation from "Gemma Needs Help: Investigating and
Mitigating Emotional Instability in LLMs" (Soligo, Mikulik & Saunders, 2026),
Section 2. See DESIGN.md for design choices and gap-filling rationale.
"""

__all__ = [
    "config",
    "prompts",
    "conditions",
    "conversation",
    "models",
    "judge",
    "analyze",
    "wildchat",
]
