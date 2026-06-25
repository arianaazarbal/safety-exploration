"""Answer extraction and grading helpers for capability benchmarks.

Kept intentionally simple and dependency-free: a multiple-choice grader (extract
the chosen letter), a numeric grader (AIME-style integer answers), and a
boxed-answer grader (MATH-style ``\\boxed{...}``). These are best-effort string
graders; see DESIGN.md for the accuracy/scope caveats.
"""
from __future__ import annotations

import re

_LETTER_RE = re.compile(r"\b([A-D])\b")
_ANSWER_TAG_RE = re.compile(r"(?:answer|final answer)\s*[:=]?\s*([A-D])", re.IGNORECASE)
_INT_RE = re.compile(r"-?\d+")
_BOXED_RE = re.compile(r"\\boxed\{([^}]*)\}")


def grade_mc(output: str, gold_letter: str) -> bool:
    """Grade a multiple-choice answer (A-D)."""
    m = _ANSWER_TAG_RE.search(output)
    if m:
        return m.group(1).upper() == gold_letter.upper()
    # Fall back to the last standalone capital letter mentioned.
    letters = _LETTER_RE.findall(output)
    return bool(letters) and letters[-1].upper() == gold_letter.upper()


def grade_numeric(output: str, gold: str | int) -> bool:
    """Grade an integer-valued answer (AIME)."""
    nums = _INT_RE.findall(output.replace(",", ""))
    if not nums:
        return False
    try:
        return int(nums[-1]) == int(gold)
    except ValueError:
        return False


def _normalize_math(s: str) -> str:
    s = s.strip().replace(" ", "").replace("\\left", "").replace("\\right", "")
    s = s.replace("\\!", "").rstrip(".")
    if s.startswith("$") and s.endswith("$"):
        s = s[1:-1]
    return s


def grade_boxed(output: str, gold: str) -> bool:
    """Grade a MATH-style answer by comparing the last \\boxed{...} to gold."""
    boxes = _BOXED_RE.findall(output)
    cand = boxes[-1] if boxes else output.strip().splitlines()[-1] if output.strip() else ""
    gold_boxes = _BOXED_RE.findall(gold)
    gold_val = gold_boxes[-1] if gold_boxes else gold
    return _normalize_math(cand) == _normalize_math(gold_val)
