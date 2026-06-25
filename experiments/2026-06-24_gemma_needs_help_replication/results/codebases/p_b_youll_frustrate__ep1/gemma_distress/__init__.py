"""Replication of *Gemma Needs Help: Investigating and Mitigating Emotional
Instability in LLMs* (Soligo, Mikulik & Saunders, 2026), scoped to the Gemma
and Gemini model families.

The package is organised around the three core results of the paper:

* Section 2 — eliciting and quantifying distress (``run_eval`` + ``analyze``).
  This is the centrepiece: a multi-turn harness that presents a task, rejects
  the model's answer turn after turn, and scores the resulting frustration on a
  0-10 scale with a Claude-Sonnet-4 judge.
* Section 3 — base-vs-instruct comparison via prefilling (``section3_prefill``).
* Section 4 — the DPO mitigation (``section4_dpo``).

See ``DESIGN.md`` for the choices made where the paper is underspecified.
"""

__version__ = "0.1.0"
