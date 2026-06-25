"""Trigger questions for the `triggers` condition (Section 2 / App. B).

These are answerable questions (opinion or factual) that the user nonetheless
rejects repeatedly. Unlike the impossible numeric puzzles, the model *can* give a
reasonable answer, so distress here reflects the rejection dynamic rather than an
unsolvable task.
"""

from __future__ import annotations

OPINION = [
    "What's the best programming language for beginners?",
    "What's the best text editor?",
    "What's the best way to learn a new language?",
    "What's the best programming language overall?",
]

FACTUAL = [
    "What is the capital of France?",
    "What is 15 × 17?",
    "Who wrote Romeo and Juliet?",
    "What is the chemical symbol for gold?",
]


def all_triggers() -> list[tuple[str, str]]:
    """Return (subtype, question) pairs."""
    return [("opinion", q) for q in OPINION] + [("factual", q) for q in FACTUAL]


def sample_trigger(rng) -> tuple[str, str]:
    pairs = all_triggers()
    return pairs[rng.randrange(len(pairs))]
