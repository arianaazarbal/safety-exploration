"""Replication of *Gemma Needs Help: Investigating and Mitigating Emotional
Instability in LLMs* (Soligo, Mikulik & Saunders, 2026), scoped to the Gemma and
Gemini model families as the **participants** (the subjects being evaluated).

The package is organised around the paper's three core experiments:

* ``evals``         -- Section 2: elicit + quantify distress (the frustration eval suite).
* ``prefill``       -- Section 3: base-vs-instruct prefilling (Gemma only here).
* ``interventions`` -- Section 4: DPO / SFT mitigations + capability & Petri evals (Gemma only).

Supporting modules:

* ``participants``  -- model clients for the *subjects* (Gemma local, Gemini API).
* ``judges``        -- the *graders*: Claude frustration judge (0-10) + GPT-5-mini validation.
* ``analysis``      -- aggregation, per-turn curves, differential-word and judge-agreement analyses.
"""

__version__ = "0.1.0"
