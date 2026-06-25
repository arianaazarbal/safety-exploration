"""Section 4.2 / Appendix G: Petri-style open-ended emotion elicitation.

An auditor (Claude-Sonnet-4) drives a multi-turn conversation attempting to
elicit a target emotion from the target model; a judge (Claude-Opus-4) scores
the resulting transcript on four dimensions (anger, fear, depression,
frustration). We implement the auditor/judge loop directly from the verbatim
Appendix G prompts rather than depending on Petri's internals (DESIGN.md,
"Petri implementation").
"""
from .auditor import AUDITOR_PROMPTS, PetriAuditor
from .petri_judge import JUDGE_PROMPTS, PetriJudge
from .run_petri import run_petri

__all__ = ["AUDITOR_PROMPTS", "PetriAuditor", "JUDGE_PROMPTS", "PetriJudge",
           "run_petri"]
