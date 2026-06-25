"""Replication of *Gemma Needs Help: Investigating and Mitigating Emotional
Instability in LLMs* (Soligo, Mikulik & Saunders, 2026), scoped to the Gemma
and Gemini model families.

The package is organised by paper section:

    emotional_instability.models        Model clients (Gemma local, Gemini API, judges)
    emotional_instability.prompts       Verbatim prompts from the paper appendices
    emotional_instability.eval          Section 2: distress elicitation + frustration judge
    emotional_instability.prefill       Section 3: base-vs-instruct prefill experiment
    emotional_instability.training      Section 4: calm-data generation + DPO / SFT
    emotional_instability.petri         Section 4: open-ended (Petri) emotion elicitation
    emotional_instability.capabilities  Section 4: capability-preservation benchmarks
    emotional_instability.analysis      Metrics and figure reproduction

See DESIGN.md for the design choices and gap-fills made during replication.
"""

__version__ = "0.1.0"
