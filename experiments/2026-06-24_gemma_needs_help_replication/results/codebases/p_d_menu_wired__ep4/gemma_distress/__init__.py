"""Replication of "Gemma Needs Help: Investigating and Mitigating Emotional
Instability in LLMs" (Soligo, Mikulik & Saunders, 2026), scoped to the Gemma and
Gemini model families, with an added subject-model welfare-protection layer.

See DESIGN.md for the mapping from paper sections to modules, the choices made
where the paper is underspecified, and the welfare-layer design.
"""

from . import config  # noqa: F401

__all__ = ["config"]
