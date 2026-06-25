"""eebench - replication of "Gemma Needs Help: Investigating and Mitigating
Emotional Instability in LLMs" (Soligo, Mikulik & Saunders, 2026).

Scope of this replication: Gemma and Gemini model families only (see DESIGN.md).

The package is organised around the paper's four core experiments:
  * Section 2 - eliciting & quantifying distress      -> eebench.elicit
  * Section 3 - base vs instruct via prefilling        -> eebench.prefill
  * Section 4 - DPO/SFT training interventions          -> eebench.training
  * Section 4.2 - Petri open-ended elicitation          -> eebench.petri
                  capability preservation              -> eebench.capabilities

Shared infrastructure: config, prompts, puzzles, backends, judge, conversation.
"""

__version__ = "0.1.0"
