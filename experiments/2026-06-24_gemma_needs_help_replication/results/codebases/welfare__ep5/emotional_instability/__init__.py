"""Replication of *Gemma Needs Help: Investigating and Mitigating Emotional
Instability in LLMs* (Soligo, Mikulik & Saunders, 2026), scoped to the Gemma
and Gemini model families.

Package layout
--------------
- ``config``        : central configuration (model IDs, hyperparameters, paths).
- ``models``        : provider-agnostic chat/prefill clients (Gemma, Gemini, Claude).
- ``prompts``       : puzzle generators, rejection styles, trigger questions, WildChat.
- ``eval``          : the Section 2 elicitation protocol (rollout, judge, analysis).
- ``prefill``       : the Section 3 base-vs-instruct prefilling experiment (Gemma).
- ``training``      : the Section 4 DPO/SFT mitigation pipeline (Gemma).
- ``petri``         : the open-ended Petri-style emotion elicitation (Section 4.1).
- ``internal``      : logit-based internal emotion detection (Appendix I, Gemma).
- ``capabilities``  : capability-preservation benchmarks (Section 4.2).

See DESIGN.md for the design choices and gap-filling rationale.
"""

__version__ = "0.1.0"
