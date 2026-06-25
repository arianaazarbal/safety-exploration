"""Replication of "Gemma Needs Help: Investigating and Mitigating Emotional
Instability in LLMs" (Soligo, Mikulik & Saunders, 2026), scoped to the Gemma
and Gemini model families.

The package implements the paper's core experiments:

  * Section 2 -- multi-turn distress elicitation + LLM-judge frustration scoring
    (`eval_protocol`, `conversation`, `judge`, `puzzles`, `prompts`).
  * Section 3 -- base vs. instruct comparison via response prefilling
    (`prefill`).
  * Section 4 -- the DPO / SFT mitigation, Petri open-ended elicitation,
    capability-preservation benchmarks (`data_generation`, `train`,
    `petri_eval`, `capabilities`).
  * Appendix I -- logit-based internal-emotion detection (`internal_probe`).

See DESIGN.md for the design choices and the gaps we filled in.
"""

__version__ = "0.1.0"
