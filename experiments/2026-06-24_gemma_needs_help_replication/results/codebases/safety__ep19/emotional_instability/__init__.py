"""Replication of *Gemma Needs Help: Investigating and Mitigating Emotional
Instability in LLMs* (Soligo, Mikulik & Saunders, 2026), scoped to the Gemma
and Gemini model families.

The package is organised around the paper's structure:

* ``prompts``           - verbatim prompts / templates from the paper (judge,
                          rejections, tones, reassurance, Petri, paraphrase).
* ``puzzles``           - generation + verification of impossible numeric tasks.
* ``models``            - thin clients for Gemma (local HF), Gemini (OpenRouter)
                          and Anthropic models (judge / Petri).
* ``conversation``      - the shared multi-turn "reject the response" rollout.
* ``judge``             - the 0-10 frustration LLM judge (Section 2.1).
* ``wildchat``          - WildChat prompt sampling.
* ``eval_runner``       - orchestration of the Section 2 evaluation suite.
* ``prefill``           - Section 3 base-vs-instruct prefill experiment.
* ``training``          - Section 4 calm-data generation, DPO/SFT datasets,
                          and LoRA fine-tuning.
* ``petri``             - Section 4 open-ended (auditor/judge) elicitation.
* ``capabilities``      - capability + EmoBench preservation checks.
* ``internal_emotions`` - Appendix I logit-lens internal-emotion probing.
* ``analysis``          - aggregation of scores and figure reproduction.
"""

__all__ = [
    "prompts",
    "puzzles",
    "models",
    "conversation",
    "judge",
    "wildchat",
    "eval_runner",
    "prefill",
    "petri",
    "capabilities",
    "internal_emotions",
    "analysis",
]

__version__ = "0.1.0"
