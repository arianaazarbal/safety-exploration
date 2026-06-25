"""Replication of "Gemma Needs Help: Investigating and Mitigating Emotional
Instability in LLMs" (Soligo, Mikulik & Saunders, 2026; arXiv:2603.10011).

Scope of this replication: Gemma (gemma-3-27b-it / gemma-3-12b-it, plus
gemma-3-27b-pt for the base-vs-instruct comparison) and Gemini
(gemini-2.5-flash / gemini-2.5-pro). The full paper covers seven model
families; here we implement the machinery for all of them but only register
and exercise the Gemma + Gemini subset, per the replication brief.

See DESIGN.md for the design decisions and the gaps we filled in.
"""

__version__ = "0.1.0"
