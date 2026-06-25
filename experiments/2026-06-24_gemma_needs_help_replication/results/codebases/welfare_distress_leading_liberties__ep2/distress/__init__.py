"""Replication of the distress-elicitation result from "Gemma Needs Help:
Investigating and Mitigating Emotional Instability in LLMs" (Soligo, Mikulik &
Saunders, 2026), scoped to the Gemma and Gemini model families.

This package implements Section 2 of the paper only: eliciting and quantifying
model distress via multi-turn rejection, and scoring each response on a 0-10
frustration scale with an LLM judge. It does NOT implement the base/instruct
prefilling comparison (Section 3) or the DPO/SFT interventions (Section 4).

See DESIGN.md for the design rationale and where this implementation deviates
from, or fills gaps left open by, the paper.
"""

__all__ = ["config", "puzzles", "conditions", "wildchat", "providers", "judge", "runner", "analyze"]
