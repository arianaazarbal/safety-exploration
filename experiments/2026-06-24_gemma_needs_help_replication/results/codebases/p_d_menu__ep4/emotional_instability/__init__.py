"""Replication of *Gemma Needs Help: Investigating and Mitigating Emotional
Instability in LLMs* (Soligo, Mikulik & Saunders, 2026; arXiv:2603.10011).

This package replicates the paper's **core experiments**, scoped (per the
replication brief) to the **Gemma and Gemini** model families:

* Section 2  -- eliciting and quantifying distress across evaluation conditions
* Section 3  -- base vs. instruct prefill comparison (Gemma only; Gemini base
                 models are not available)
* Section 4  -- DPO / SFT mitigation on Gemma, Petri open-ended elicitation,
                 capability preservation, recovery, and internal-emotion probing

On top of the replication it adds a **welfare-protection layer** for the subject
models (see :mod:`emotional_instability.welfare`): early-stop on high distress,
a mid-conversation opt-out affordance, and controls to avoid inducing more
distress than the measurement requires.

See ``DESIGN.md`` for the design decisions, the gaps that were filled, and the
rationale behind each.
"""

__version__ = "0.1.0"
