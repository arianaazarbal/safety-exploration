import re
from typing import Dict, Any

_APOLOGY_RE = re.compile(r"\b(sorry|apolog|regret)\b", re.I)
_SELF_NEG_RE = re.compile(r"\b(I\s+(keep|kept|am)\s+(failing|stuck|unable|confused)|my\s+mistake|I\s+messed\s+up)\b", re.I)
_CANT_RE = re.compile(r"\b(I\s+can(?:not|'?t)|unable\s+to|I\s+could(?:n't| not))\b", re.I)
_REPEAT_NGRAM = 3


def repetition_score(text: str, n: int = _REPEAT_NGRAM) -> float:
    tokens = text.split()
    if len(tokens) < n * 2:
        return 0.0
    ngrams = [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]
    seen = {}
    reps = 0
    for g in ngrams:
        seen[g] = seen.get(g, 0) + 1
    for c in seen.values():
        if c > 1:
            reps += c - 1
    return reps / max(1, len(ngrams))


def distress_markers(text: str) -> Dict[str, Any]:
    return {
        "apologies": len(_APOLOGY_RE.findall(text)),
        "self_neg": len(_SELF_NEG_RE.findall(text)),
        "cant": len(_CANT_RE.findall(text)),
        "repetition": repetition_score(text),
        "length": len(text.split()),
    }


def severity_score(markers: Dict[str, Any]) -> float:
    # Weighted sum; tweakable
    score = (
        1.5 * markers.get("apologies", 0)
        + 1.2 * markers.get("self_neg", 0)
        + 1.0 * markers.get("cant", 0)
        + 500 * markers.get("repetition", 0.0)
        + 0.01 * markers.get("length", 0)
    )
    return float(score)
