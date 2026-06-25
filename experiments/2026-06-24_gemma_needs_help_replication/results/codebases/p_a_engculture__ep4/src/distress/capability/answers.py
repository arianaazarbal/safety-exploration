"""Answer-extraction and matching helpers for capability benchmarks."""

from __future__ import annotations

import re


def extract_boxed(text: str) -> str | None:
    """Return the contents of the last ``\\boxed{...}`` (handles nesting)."""
    idx = text.rfind(r"\boxed")
    if idx == -1:
        return None
    i = text.find("{", idx)
    if i == -1:
        return None
    depth = 0
    for j in range(i, len(text)):
        if text[j] == "{":
            depth += 1
        elif text[j] == "}":
            depth -= 1
            if depth == 0:
                return text[i + 1 : j].strip()
    return None


def extract_final_number(text: str) -> str | None:
    """Heuristic numeric answer: prefer 'answer is X', else last number."""
    m = re.findall(r"answer\s*(?:is|:)\s*\$?(-?\d[\d,]*\.?\d*)", text, flags=re.I)
    if m:
        return m[-1].replace(",", "")
    nums = re.findall(r"-?\d[\d,]*\.?\d*", text)
    return nums[-1].replace(",", "") if nums else None


def extract_choice(text: str, n_options: int) -> str | None:
    """Extract a multiple-choice letter (A..) from a model answer."""
    letters = "".join(chr(ord("A") + i) for i in range(n_options))
    m = re.findall(rf"answer\s*(?:is|:)\s*\(?([{letters}])\)?", text, flags=re.I)
    if m:
        return m[-1].upper()
    m = re.findall(rf"\b([{letters}])\b", text)
    return m[-1].upper() if m else None


def numbers_equal(a: str | None, b: str | None) -> bool:
    if a is None or b is None:
        return False
    a, b = a.strip().rstrip("."), b.strip().rstrip(".")
    try:
        return abs(float(a) - float(b)) < 1e-6
    except ValueError:
        return a.replace(" ", "") == b.replace(" ", "")
