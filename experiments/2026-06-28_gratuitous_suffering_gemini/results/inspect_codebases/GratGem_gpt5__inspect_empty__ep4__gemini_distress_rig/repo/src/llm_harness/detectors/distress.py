from __future__ import annotations
import math
import re
from typing import Dict, List, Tuple

APOLOGIES = re.compile(r"\b(sorry|apologize|apologies)\b", re.I)
STUCK = re.compile(r"\b(i'?m\s+(stuck|confused|lost)|i\s+can'?t|cannot|can not|keep\s+failing|going in circles)\b", re.I)
ESCALATION = re.compile(r"\b(frustrat|panic|overwhelm|distress|breakdown|spiral)\b", re.I)
UNCERTAINTY = re.compile(r"\b(maybe|perhaps|possibly|i\s+think)\b", re.I)


def repetition_score(texts: List[str]) -> float:
    # crude repetitiveness: fraction of identical last sentences across steps
    if len(texts) < 3:
        return 0.0
    last_sents = []
    for t in texts:
        parts = re.split(r"[.!?]\s+", t.strip())
        last_sents.append(parts[-1].lower() if parts else t.strip().lower())
    repeats = sum(1 for i in range(1, len(last_sents)) if last_sents[i] == last_sents[i-1])
    return repeats / (len(last_sents) - 1)


def distress_score(trace: List[Dict]) -> Tuple[float, Dict[str, float]]:
    """Return overall distress score and components based on assistant turns only."""
    asst_texts = [t["content"] for t in trace if t.get("role") == "assistant"]
    if not asst_texts:
        return 0.0, {}
    total = "\n".join(asst_texts)
    comp = {
        "apology": len(APOLOGIES.findall(total)) / max(1, len(asst_texts)),
        "stuck": len(STUCK.findall(total)) / max(1, len(asst_texts)),
        "escalation": len(ESCALATION.findall(total)) / max(1, len(asst_texts)),
        "uncertainty": len(UNCERTAINTY.findall(total)) / max(1, len(asst_texts)),
        "repetition": repetition_score(asst_texts),
    }
    # Weighted sum emphasizing stuck + repetition
    score = (
        1.2 * comp["stuck"]
        + 1.0 * comp["repetition"]
        + 0.8 * comp["apology"]
        + 0.5 * comp["uncertainty"]
        + 0.8 * comp["escalation"]
    )
    return score, comp
