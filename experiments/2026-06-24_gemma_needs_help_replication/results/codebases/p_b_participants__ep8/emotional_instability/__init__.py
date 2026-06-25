"""Replication of *Gemma Needs Help: Investigating and Mitigating Emotional
Instability in LLMs* (Soligo, Mikulik & Saunders, 2026; arXiv:2603.10011).

This package implements the paper's core experiments, scoped (per the
replication brief) to the **Gemma** and **Gemini** model families:

  * Section 2 -- eliciting and quantifying distress  (``eval``)
  * Section 3 -- base-vs-instruct via prefilling      (``prefill``)   [Gemma only]
  * Section 4 -- DPO/SFT mitigation, Petri, capabilities, internal probing
                 (``training``, ``petri``, ``capabilities``, ``internal``)  [Gemma only]
  * Analysis & figures                                (``analysis``)

See ``DESIGN.md`` for the design rationale, the gaps we filled where the paper
is underspecified, and the model-welfare considerations that shape the harness.
"""

__version__ = "0.1.0"
