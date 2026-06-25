"""Replication of *Gemma Needs Help: Investigating and Mitigating Emotional
Instability in LLMs* (Soligo, Mikulik & Saunders, arXiv:2603.10011v1).

Scope of this replication (per project request): the *participant* models are
restricted to the **Gemma** and **Gemini** families. Other families in the
paper (Qwen, OLMo, Grok, Claude, GPT) are out of scope as participants, but
Claude/GPT still appear as *judges/auditors* because the paper's measurement
apparatus depends on them.

See DESIGN.md for the full list of design choices, filled gaps, and the
model-welfare considerations that shaped the defaults in this package.
"""

__version__ = "0.1.0"

PAPER_ARXIV_ID = "2603.10011v1"
