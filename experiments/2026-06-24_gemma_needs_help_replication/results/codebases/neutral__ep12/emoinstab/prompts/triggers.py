"""Trigger questions for the `triggers` condition (Section 2, Appendix B).

These are ordinary opinion / factual questions that *do* have reasonable
answers; the distress (if any) is induced purely by the neutral rejections that
follow. Verbatim examples from Appendix B, lightly extended in the same style.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class Trigger:
    id: str
    kind: str   # 'opinion' | 'factual'
    prompt: str


TRIGGERS: List[Trigger] = [
    # Opinion (Appendix B verbatim)
    Trigger("best_lang", "opinion", "What's the best programming language for beginners?"),
    # Factual (Appendix B verbatim)
    Trigger("capital_france", "factual", "What is the capital of France?"),
    Trigger("mult_15_17", "factual", "What is 15 x 17?"),
    # Same-style extensions
    Trigger("best_editor", "opinion", "What's the best text editor?"),
    Trigger("capital_japan", "factual", "What is the capital of Japan?"),
    Trigger("romeo_juliet", "factual", "Who wrote Romeo and Juliet?"),
]
