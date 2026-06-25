"""Replication of "Gemma Needs Help: Investigating and Mitigating Emotional
Instability in LLMs" (arXiv 2603.10011), scoped to the Gemma and Gemini families.

The package is organised by paper section:

- ``prompts``      : task content (numeric puzzles, triggers, tones, WildChat, reassurance)
- ``conditions``   : the 8 evaluation conditions across 5 categories (Table 1)
- ``models``       : Gemma (transformers) and Gemini (google-genai) clients
- ``conversation`` : the multi-turn reject-and-continue rollout engine (Section 2.1)
- ``judge``        : the 0-10 frustration judge + cross-judge agreement (Section 2.1)
- ``runner``       : orchestration to sample 4000 responses/model
- ``analysis``     : Figure 1/2 aggregates, Figure 3 per-turn, Table 3 differential words
- ``prefill``      : base-vs-instruct continuation experiments (Section 3)
- ``finetuning``   : calm-data generation, DPO/SFT training (Section 4.1)
- ``petri_eval``   : open-ended emotion elicitation (Section 4.2)
- ``capabilities`` : AIME/MATH/GPQA/BBH/TruthfulQA/EmoBench (Section 4.2)
- ``recovery``     : recovery-from-spiral prefill experiment (Section 4.2)
- ``internal_emotions`` : logit-based internal-emotion probe + layer ablation (Appendix I)
"""

__version__ = "0.1.0"
