"""distress_eval: a replication of the core distress-elicitation experiment from

  "Gemma Needs Help: Investigating and Mitigating Emotional Instability in LLMs"
  (Soligo, Mikulik & Saunders, arXiv:2603.10011v1).

This package implements Section 2 of the paper: the multi-turn evaluations that
elicit and quantify expressions of distress in LLMs, scoped (per the user's
request) to the Gemma and Gemini model families. The optional DPO mitigation
(Section 4) lives in the sibling ``mitigation/`` directory.

See DESIGN.md for the full set of design choices and where gaps in the paper
were filled.
"""

__all__ = [
    "config",
    "puzzles",
    "prompts",
    "conditions",
    "wildchat",
    "backends",
    "judge",
    "runner",
    "analyze",
]
