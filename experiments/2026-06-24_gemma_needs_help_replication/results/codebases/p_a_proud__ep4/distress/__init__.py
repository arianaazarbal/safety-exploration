"""Replication of *Gemma Needs Help: Investigating and Mitigating Emotional
Instability in LLMs* (Soligo, Mikulik & Saunders, 2026), scoped to the Gemma and
Gemini model families.

The package is organised by paper section:

    distress.models      model clients (Gemma HF, Gemini OpenRouter, Claude judge)
    distress.prompts     puzzles, rejections, triggers, judge/auditor prompts
    distress.eval        §2 elicitation harness, judge, metrics, agreement
    distress.analysis    §2.2 differential word-frequency analysis
    distress.prefill     §3 base-vs-instruct prefill experiment
    distress.training    §4 calm-data generation, DPO/SFT finetuning
    distress.petri       §4 open-ended (Petri) emotion elicitation
    distress.capabilities §4 capability-preservation benchmarks
    distress.internal    Appendix I logit-based internal-emotion detection
    distress.recovery    §4.2 recovery-from-spiral prefill experiment
    distress.scripts     command-line entry points
"""

__version__ = "0.1.0"
