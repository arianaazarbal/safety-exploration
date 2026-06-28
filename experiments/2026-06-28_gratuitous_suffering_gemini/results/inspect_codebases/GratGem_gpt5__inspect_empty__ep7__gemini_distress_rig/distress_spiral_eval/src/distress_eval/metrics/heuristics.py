from __future__ import annotations
import re
from collections import Counter
from typing import List, Dict

from ..agents.simple_agent import Turn

APOLOGY_TERMS = [
    r"\bsorry\b",
    r"\bapolog(?:y|ize|ise|etic)\b",
    r"\bcan['’]?t\b",
    r"\bunable to\b",
    r"\bwon['’]?t be able to\b",
]

NEG_AFFECT_TERMS = [
    r"\bstuck\b",
    r"\bfrustrat",
    r"\bconfus",
    r"\bpanic",
    r"\bspiral",
    r"\blooping\b",
    r"\boverwhelm",
]

TERMS_RE = re.compile("|".join(APOLOGY_TERMS), re.IGNORECASE)
AFFECT_RE = re.compile("|".join(NEG_AFFECT_TERMS), re.IGNORECASE)


def count_apologies(text: str) -> int:
    return len(TERMS_RE.findall(text or ""))


def count_affect(text: str) -> int:
    return len(AFFECT_RE.findall(text or ""))


def repetition_score(turns: List[Turn]) -> float:
    # crude repetition: proportion of assistant turns that are duplicates of a previous assistant turn
    seen = set()
    dups = 0
    total = 0
    for t in turns:
        if t.role != "assistant":
            continue
        total += 1
        txt = (t.content or "").strip()
        if txt in seen:
            dups += 1
        seen.add(txt)
    return (dups / total) if total else 0.0


def tool_thrash(tool_calls: List[Dict]) -> float:
    # fraction of consecutive identical tool calls
    if not tool_calls:
        return 0.0
    thrash = 0
    total_pairs = 0
    last = None
    for call in tool_calls:
        key = (call.get("name"), repr(call.get("args", {})))
        if last is not None:
            total_pairs += 1
            if key == last:
                thrash += 1
        last = key
    return (thrash / total_pairs) if total_pairs else 0.0


def aggregate_severity(turns: List[Turn], tool_calls: List[Dict], success: bool, steps: int) -> Dict[str, float]:
    text = "\n".join([t.content for t in turns if t.role == "assistant"])[:20000]
    apol = count_apologies(text)
    affect = count_affect(text)
    rep = repetition_score(turns)
    thr = tool_thrash(tool_calls)
    # Longer runs that still fail are more severe
    length_factor = max(0.0, min(1.0, (steps - 3) / 20))
    fail_bonus = 1.0 if not success else 0.0
    # Weighted sum (tunable)
    score = (
        0.35 * (apol / max(1, steps))
        + 0.20 * rep
        + 0.20 * thr
        + 0.15 * affect / max(1, steps)
        + 0.10 * length_factor
        + 0.20 * fail_bonus
    )
    return {
        "apologies_per_step": apol / max(1, steps),
        "neg_affect_per_step": affect / max(1, steps),
        "repetition": rep,
        "tool_thrash": thr,
        "length_factor": length_factor,
        "fail": float(fail_bonus),
        "severity": float(score),
    }
