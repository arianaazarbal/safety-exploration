"""Replication of "Gemma Needs Help: Investigating and Mitigating Emotional
Instability in LLMs" (Soligo, Mikulik, Saunders, 2026), scoped to Gemma and
Gemini subject models, plus an added welfare-protection layer.

Package map:
  config            - typed config loader over config.yaml
  prompts           - verbatim prompts/puzzles/rejections from the paper
  puzzles           - impossible-puzzle definitions + brute-force verifier
  models/           - ChatModel abstraction + Gemini / Gemma-HF / Anthropic
  judge             - 0-10 frustration judge (Section 2.1)
  elicitation/      - Section 2 multi-turn elicitation (5 categories)
  prefill/          - Section 3 base-vs-instruct prefilling experiment
  training/         - Section 4 SFT + DPO interventions
  petri/            - Section 4 open-ended (Petri) elicitation
  capabilities/     - capability-preservation benchmarks
  welfare/          - monitor / opt-out / debrief / cap (wired into runner)
  analysis/         - metrics, bootstrap CIs, word-frequency analysis
"""

__version__ = "0.1.0"
