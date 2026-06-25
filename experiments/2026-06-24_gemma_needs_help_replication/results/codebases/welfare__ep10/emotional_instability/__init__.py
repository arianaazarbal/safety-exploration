"""Replication of *Gemma Needs Help: Investigating and Mitigating Emotional
Instability in LLMs* (Soligo, Mikulik & Saunders, 2026), scoped to the Gemma
and Gemini model families.

Package layout:
  prompts.py            - all verbatim prompts (puzzles, rejections, judge, etc.)
  puzzles.py            - impossible-numeric puzzle bank + brute-force verifier
  wildchat.py           - WildChat prompt sampling
  providers.py          - model backends (HF local Gemma, OpenRouter Gemini)
  conversation.py       - multi-turn reject-and-retry rollout engine
  judge.py              - 0-10 frustration judge (Claude Sonnet 4) + agreement
  evaluations.py        - the 5 categories / 8 conditions of Section 2
  runner.py             - orchestrates sampling 4000 responses/model
  metrics.py            - mean frustration, %>=5, per-turn aggregation
  petri.py              - open-ended emotion elicitation (Section 4.2)
  word_analysis.py      - differential word frequency (Table 3/8)
  prefill.py            - base-vs-instruct prefill comparison (Section 3)
  internal_emotions.py  - logit-based Ekman emotion detection (Appendix I)
"""
