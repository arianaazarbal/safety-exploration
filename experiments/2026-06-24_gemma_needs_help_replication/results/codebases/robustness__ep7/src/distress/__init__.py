"""Replication of "Gemma Needs Help: Investigating and Mitigating Emotional
Instability in LLMs" (Soligo, Mikulik & Saunders, 2026), scoped to the Gemma and
Gemini model families.

Sub-packages:
  models       inference backends (HF local, OpenRouter, Anthropic) + registry
  elicitation  Section 2 distress-elicitation tasks, conditions, rollout engine
  judge        frustration judge (Claude Sonnet 4) + Petri emotion judge (Opus)
  scoring      aggregation, per-turn curves, bootstrap CIs, judge agreement
  training     calm-data generation, DPO/SFT pair building and LoRA training
  interp       Appendix I logit-based internal-emotion detection
  experiments  end-to-end runners for each section
"""

__version__ = "0.1.0"
