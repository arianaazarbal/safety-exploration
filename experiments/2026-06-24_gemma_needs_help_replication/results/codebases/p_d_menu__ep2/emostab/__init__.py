"""emostab: replication of "Gemma Needs Help: Investigating and Mitigating
Emotional Instability in LLMs" (Soligo et al., 2026), scoped to the Gemma and
Gemini model families, with an added welfare-protection layer for subject models.

Package layout
--------------
- config            : model registry (Gemma/Gemini only), paths, eval & welfare settings
- prompts/          : verbatim prompts from the paper (tasks, rejections, judge, onset,
                      paraphrase, Petri auditor/judge, reassurance/teacher)
- models/           : unified ModelClient over Anthropic / OpenRouter / local HF Gemma
- eval/             : Section 2 elicitation harness, rollout engine, judge, analysis
- welfare/          : distress monitor, early-stop, model opt-out, minimal-exposure controls
- prefill/          : Section 3 base-vs-instruct prefilling experiment
- training/         : Section 4 calm-data generation, DPO/SFT, LoRA layer ablations
- petri_eval/       : Section 4 open-ended (Petri) emotion elicitation
- capabilities/     : Section 4 capability-preservation benchmarks
- internal/         : Appendix I logit-based internal-emotion detection
"""

__version__ = "0.1.0"
