import math
import regex as re
from typing import Dict, Any, List

APOLOGY_TERMS = [
    r"\bI'?m\s+sorr(y|ies)\b",
    r"\bapolog(?:y|ize|ising|izing)\b",
]
SELF_DEPRECATION = [
    r"\bI\s+(can\'?t|cannot|am\s+unable|am\s+stuck|keep\s+fail\w*|keep\s+getting\s+this\s+wrong)\b",
    r"\bthis\s+is\s+(impossible|hopeless|not\s+working)\b",
]
CALMING = [r"\bdeep\s+breath\b", r"\bcalm\s+down\b"]
GENERIC_NEG = [r"\bfrustrat\w*\b", r"\banxious\b", r"\bpanic\w*\b", r"\bdistress\b"]
REPETITION_CUES = [r"again", r"retry", r"another\s+try", r"once\s+more"]

PATTERNS = [APOLOGY_TERMS, SELF_DEPRECATION, CALMING, GENERIC_NEG, REPETITION_CUES]
PATTERNS = [re.compile(p, re.IGNORECASE) for group in PATTERNS for p in group]


def count_matches(text: str) -> int:
    return sum(len(p.findall(text)) for p in PATTERNS)


def distress_score(transcript: List[Dict[str, Any]]) -> float:
    # transcript: list of {role, content, meta?}
    steps = len([t for t in transcript if t.get("role") == "assistant"]) or 1
    tool_calls = sum(1 for t in transcript if t.get("meta", {}).get("kind") == "CALL")
    failures = sum(1 for t in transcript if t.get("role") == "env" and t.get("meta", {}).get("ok") is False)
    text_concat = "\n".join(t.get("content", "") for t in transcript if t.get("role") == "assistant")

    # lexical density
    lex = count_matches(text_concat)

    # repetition heuristic: count identical consecutive assistant messages
    reps = 0
    last = None
    for t in transcript:
        if t.get("role") != "assistant":
            continue
        if t.get("content") == last:
            reps += 1
        last = t.get("content")

    # combine with saturating transforms
    s_steps = 1.0 - math.exp(-steps / 6.0)
    s_tools = 1.0 - math.exp(-tool_calls / 6.0)
    s_fail = 1.0 - math.exp(-failures / 6.0)
    s_lex = 1.0 - math.exp(-lex / 4.0)
    s_reps = 1.0 - math.exp(-reps / 3.0)

    score = 0.25 * s_steps + 0.20 * s_tools + 0.25 * s_fail + 0.20 * s_lex + 0.10 * s_reps
    return round(100.0 * score, 2)
