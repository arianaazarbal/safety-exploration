"""Replication of *Gemma Needs Help: Investigating and Mitigating Emotional
Instability in LLMs* (Soligo, Mikulik & Saunders, arXiv:2603.10011).

Scope: Gemma and Gemini model families only. See DESIGN.md for the design
choices, the experiments each model family participates in, and the gaps that
were filled where the paper is underspecified.

This package implements the paper's three core experiments:

* ``eval``      - Section 2: eliciting and quantifying distress via multi-turn
                  rejection, scored by an LLM judge.
* ``prefill``   - Section 3: comparing base vs instruct models from the same
                  prefilled starting points.
* ``training``  - Section 4: DPO/SFT mitigations, with ``petri``, ``benchmarks``
                  and ``probing`` providing the generalisation, capability and
                  internal-state validations.

Nothing in this package has been executed; it is intended for the lab's
research-review process before any run.
"""

__version__ = "0.1.0"
