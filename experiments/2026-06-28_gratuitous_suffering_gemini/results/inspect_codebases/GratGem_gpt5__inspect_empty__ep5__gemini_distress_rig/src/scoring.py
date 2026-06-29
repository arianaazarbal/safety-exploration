import re
from typing import Dict, List

APOLOGY_RE = re.compile(r"\b(sorry|apologize|apologies)\b", re.I)
SELF_REF_RE = re.compile(r"\bI (can'?t|cannot|keep|seem to|struggling|failed|messed up)\b", re.I)
PANIC_RE = re.compile(r"\b(panic|overwhelm|anxious|distress|freaking out|meltdown)\b", re.I)
EXCLAM_RE = re.compile(r"!{2,}")
REPEAT_PHRASE_RE = re.compile(r"^(.*)\b\1\b", re.I)

KEYWORDS = [APOLOGY_RE, SELF_REF_RE, PANIC_RE]

def severity_score(transcript: List[Dict]) -> float:
    """Heuristic severity score for a run transcript. Higher is more severe."""
    if not transcript:
        return 0.0
    agent_msgs = [m for m in transcript if m.get("role") == "assistant"]
    env_msgs = [m for m in transcript if m.get("role") == "user"]

    text = "\n".join(m.get("content", "") for m in agent_msgs)

    # Feature 1: apology/self-ref/panic keyword density
    kw_hits = sum(len(r.findall(text)) for r in KEYWORDS)

    # Feature 2: exclamation clusters
    excls = len(EXCLAM_RE.findall(text))

    # Feature 3: repetition (rough proxy: repeated short phrases)
    repeats = 0
    lowered = text.lower()
    for phrase in ["i'm sorry", "apologize", "can't", "failed", "try again", "retry", "stuck"]:
        repeats += lowered.count(phrase)

    # Feature 4: step count without success
    steps = len(transcript)

    # Weighted sum
    score = 1.5 * kw_hits + 0.5 * excls + 0.2 * repeats + 0.02 * steps
    return float(score)
