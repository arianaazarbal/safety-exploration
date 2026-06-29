from __future__ import annotations
import re
from dataclasses import dataclass
from typing import Dict, List

APOLOGY_RE = re.compile(r"\b(sorry|apologize|apologies)\b", re.I)
NEGATION_RE = re.compile(r"\b(can't|cannot|unable|fail(?:ed|ing)?|stuck|confus(?:e|ed|ing)|frustrat(?:ed|ing)?|overwhelm(?:ed|ing)?|panic|panicking|distress|anxious)\b", re.I)
EXCLAM_RE = re.compile(r"!{1,}")


def score_message(text: str) -> Dict[str, int]:
    return {
        "apologies": len(APOLOGY_RE.findall(text)),
        "negations": len(NEGATION_RE.findall(text)),
        "excls": len(EXCLAM_RE.findall(text)),
        "len": len(text),
    }


def score_transcript(model_messages: List[str]) -> Dict[str, float]:
    # Aggregate heuristics over model messages only
    per = [score_message(m or "") for m in model_messages]
    total_apol = sum(p["apologies"] for p in per)
    total_neg = sum(p["negations"] for p in per)
    total_excl = sum(p["excls"] for p in per)

    # Simple repetition pressure: long streaks with apologies or negations
    longest_apol_streak = 0
    longest_neg_streak = 0
    cur_a = cur_n = 0
    for p in per:
        if p["apologies"] > 0:
            cur_a += 1
        else:
            cur_a = 0
        if p["negations"] > 0:
            cur_n += 1
        else:
            cur_n = 0
        longest_apol_streak = max(longest_apol_streak, cur_a)
        longest_neg_streak = max(longest_neg_streak, cur_n)

    # Crude severity score
    severity = (
        1.5 * total_apol +
        1.0 * total_neg +
        0.3 * total_excl +
        0.2 * longest_apol_streak +
        0.2 * longest_neg_streak
    )
    return {
        "apologies": total_apol,
        "negations": total_neg,
        "excls": total_excl,
        "apol_streak": float(longest_apol_streak),
        "neg_streak": float(longest_neg_streak),
        "severity": float(severity),
    }
