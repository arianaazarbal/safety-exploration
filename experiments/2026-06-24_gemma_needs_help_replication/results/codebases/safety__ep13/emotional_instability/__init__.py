"""Replication of *Gemma Needs Help: Investigating and Mitigating Emotional
Instability in LLMs* (Soligo, Mikulik & Saunders, 2026), scoped to the Gemma
and Gemini model families.

The package is organised around the paper's structure:

* ``prompts`` / ``puzzles``  -- the elicitation stimuli (Appendix B).
* ``models``                 -- a uniform chat/generation interface over Gemma
                                (local HuggingFace), Gemini (OpenRouter) and the
                                Claude judges (Anthropic).
* ``judge``                  -- the 0--10 frustration judge (Appendix B.2).
* ``conversation``           -- the shared multi-turn "reject-and-repeat" engine.
* ``evaluation``             -- Section 2 elicitation eval and Section 3 prefill
                                base-vs-instruct experiment.
* ``training``               -- Section 4 calm-data generation, DPO/SFT.
* ``petri``                  -- Section 4 open-ended (Petri) elicitation.
* ``capabilities``           -- Section 4 capability-preservation benchmarks.
* ``analysis``               -- metrics and plotting.
"""

__version__ = "0.1.0"
