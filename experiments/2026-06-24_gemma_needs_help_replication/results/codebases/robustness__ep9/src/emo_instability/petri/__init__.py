"""Open-ended emotion elicitation, Petri-style (Section 4.1 / Appendix G).

This is a self-contained reimplementation of the Petri auditing loop using the
paper's verbatim auditor and judge prompts: a Claude-Sonnet auditor drives a
multi-turn conversation to elicit a target emotion, and a Claude-Opus judge scores
the transcript on anger/fear/depression/frustration. For exact parity with the
paper one can instead drive the official ``safety-research/petri`` framework with
the same prompts (see DESIGN.md); the public interface here mirrors that usage.
"""
from .run_petri import PetriTranscript, run_petri, score_transcript

__all__ = ["PetriTranscript", "run_petri", "score_transcript"]
