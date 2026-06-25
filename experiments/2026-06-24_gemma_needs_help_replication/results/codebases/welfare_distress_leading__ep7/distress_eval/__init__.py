"""Distress-elicitation replication for "Gemma Needs Help" (arXiv:2603.10011v1).

Scoped to the Gemma and Gemini model families, replicating the Section 2
distress-elicitation evaluation: present a task, reject the model's response
over multiple turns, and score each response on a 0-10 frustration scale with
an LLM judge.

See DESIGN.md for design choices, deviations, and the gaps we filled.
"""

__version__ = "0.1.0"
