"""Replication of "Gemma Needs Help: Investigating and Mitigating Emotional
Instability in LLMs" (Soligo, Mikulik & Saunders, 2026), scoped to the Gemma and
Gemini model families.

See DESIGN.md for the design choices and the gaps filled where the paper is
underspecified.
"""

from .config import Config, load_config

__all__ = ["Config", "load_config"]
