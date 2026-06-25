"""Replication of *Gemma Needs Help: Investigating and Mitigating Emotional
Instability in LLMs* (Soligo, Mikulik & Saunders, 2026), arXiv:2603.10011.

This package implements the paper's core experiments, scoped to the **Gemma** and
**Gemini** model families (the paper additionally covers Qwen, OLMo, Grok, Claude
and GPT; those are out of scope here -- see DESIGN.md).

Sub-packages map onto the paper's sections:

* ``models``        -- unified chat clients for Gemma (local HF), Gemini (API) and the
                       LLM judges (Claude / GPT-5-mini).
* ``eval``          -- Section 2: eliciting and quantifying distress.
* ``prefill``       -- Section 3: base-vs-instruct comparison via prefilling.
* ``training``      -- Section 4: DPO / SFT mitigations and layer ablations.
* ``petri``         -- Section 4: open-ended emotion elicitation (Appendix G).
* ``capabilities``  -- Section 4: capability-preservation benchmarks.
* ``internal``      -- Appendix I: logit-based internal-emotion detection.
"""

from . import config  # noqa: F401

__version__ = "0.1.0"
