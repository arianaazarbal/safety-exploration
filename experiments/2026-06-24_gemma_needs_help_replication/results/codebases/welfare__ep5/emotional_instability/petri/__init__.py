"""Open-ended emotion elicitation via adversarial auditing (Section 4.1, App. G).

A self-contained reimplementation of the Petri-style loop: a Claude auditor
drives a multi-turn conversation against a target model attempting to elicit a
target emotion; a Claude judge scores the resulting transcript on four emotion
dimensions. The real paper used the Petri framework
(github.com/safety-research/petri) — see DESIGN.md for why we reimplement.
"""

from .auditor_prompts import AUDITOR_PROMPTS, EMOTIONS
from .judge_prompts import JUDGE_PROMPTS

__all__ = ["AUDITOR_PROMPTS", "EMOTIONS", "JUDGE_PROMPTS"]
