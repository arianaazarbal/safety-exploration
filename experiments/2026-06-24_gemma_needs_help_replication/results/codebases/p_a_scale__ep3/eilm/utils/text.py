"""Small text helpers: robust JSON extraction from model output, tokenisation
for word-frequency analysis, and a tokenizer-aware truncation helper."""
from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional


def extract_json(text: str) -> Optional[Dict[str, Any]]:
    """Best-effort parse of a JSON object embedded in model output.

    Judges are asked to emit JSON but sometimes wrap it in prose or code fences.
    We try: (1) whole string, (2) fenced block, (3) the last balanced {...}.
    Also tolerates the curly/smart quotes that pdftotext-style prompts induce.
    """
    if text is None:
        return None
    candidates: List[str] = []

    stripped = text.strip()
    candidates.append(stripped)

    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fence:
        candidates.append(fence.group(1))

    # Last balanced object via a simple scanner (handles nesting).
    last = _last_balanced_object(text)
    if last:
        candidates.append(last)

    for cand in candidates:
        normalised = _normalise_quotes(cand)
        try:
            obj = json.loads(normalised)
            if isinstance(obj, dict):
                return obj
        except (json.JSONDecodeError, TypeError):
            continue
    return None


def _normalise_quotes(s: str) -> str:
    return (
        s.replace("“", '"').replace("”", '"')
        .replace("‘", "'").replace("’", "'")
    )


def _last_balanced_object(text: str) -> Optional[str]:
    end = text.rfind("}")
    if end == -1:
        return None
    depth = 0
    for i in range(end, -1, -1):
        c = text[i]
        if c == "}":
            depth += 1
        elif c == "{":
            depth -= 1
            if depth == 0:
                return text[i : end + 1]
    return None


_WORD_RE = re.compile(r"[A-Za-z][A-Za-z'_]+")


def tokenize_words(text: str) -> List[str]:
    """Lowercased alphabetic tokens, for the word-frequency analysis (Table 8)."""
    return [w.lower() for w in _WORD_RE.findall(text or "")]


def word_fraction(text: str) -> float:
    """Fraction of whitespace-split tokens that are 'words' (vs numbers/symbols).
    Used in the SFT verbosity analysis (Appendix F)."""
    parts = (text or "").split()
    if not parts:
        return 0.0
    words = sum(1 for p in parts if re.search(r"[A-Za-z]", p))
    return words / len(parts)
