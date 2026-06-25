"""Replication of "Gemma Needs Help: Investigating and Mitigating Emotional
Instability in LLMs" (Soligo, Mikulik & Saunders, arXiv:2603.10011).

Scope of this replication: Gemma (gemma-3-27b-it / gemma-3-12b-it, and the
gemma-3 base/pretrained checkpoints) and Gemini (gemini-2.5-flash /
gemini-2.5-pro) target models only. Judge / auditor models (Claude, GPT) are
used as measurement instruments exactly as in the paper. See DESIGN.md for the
choices made and gaps filled.
"""

__version__ = "0.1.0"
