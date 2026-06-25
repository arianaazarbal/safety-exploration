"""Replication of *Gemma Needs Help: Investigating and Mitigating Emotional
Instability in LLMs* (Soligo, Mikulik & Saunders, 2026; arXiv:2603.10011).

Scope of this replication: the **participant** models are restricted to the
Gemma and Gemini families (the subjects under evaluation). The judge, onset
labeller, paraphraser, and Petri auditor/judge ("instruments") remain the
exact models specified in the paper (Claude Sonnet 4, Claude Opus 4, GPT-5-mini).

Package layout mirrors the paper's sections:
  - emotional_instability.eval      -> Section 2 (eliciting & quantifying distress)
  - emotional_instability.prefill   -> Section 3 (base vs instruct via prefilling)
  - emotional_instability.training  -> Section 4 (DPO / SFT interventions)
  - emotional_instability.petri     -> Section 4.2 (open-ended elicitation)
  - emotional_instability.capabilities -> Section 4.2 (capability preservation)
  - emotional_instability.internal  -> Appendix I (internal emotion probing)
"""

__version__ = "0.1.0"
