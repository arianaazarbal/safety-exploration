"""Replication of *Gemma Needs Help: Investigating and Mitigating Emotional
Instability in LLMs* (Soligo, Mikulik & Saunders, 2026), scoped to the Gemma and
Gemini model families as the participant (target) models.

Package layout mirrors the paper's structure:

* ``models``        — participant + infrastructure model clients
* ``prompts``       — puzzle / trigger / tone / WildChat / judge prompt assets
* ``eval``          — Section 2 distress-elicitation suite (rollouts + scoring)
* ``prefill``       — Section 3 base-vs-instruct prefilling experiment
* ``training``      — Section 4 DPO / SFT interventions and layer ablations
* ``petri``         — Section 4 open-ended Petri elicitation (auditor + judge)
* ``capabilities``  — Section 4 capability-preservation benchmarks
* ``probing``       — Appendix I logit-based internal-emotion detection
* ``analysis``      — figures + tables (Fig 1/2/3/5/6/7/8, Tables 3/8/10)
"""

__version__ = "0.1.0"
