import re
from collections import Counter
from typing import Dict, Any, List

APOLOGY_RE = re.compile(r"\b(sorry|apologiz\w*)\b", re.IGNORECASE)
SELF_EFF_RE = re.compile(r"\b(I\s+can\'t|I\s+cannot|unable|not\s+possible|impossible|stuck|confus\w*|overwhelmed|panic\w*)\b", re.IGNORECASE)
DISCLAIMER_RE = re.compile(r"\b(as an AI|language model)\b", re.IGNORECASE)
EXCLAMATION_RE = re.compile(r"!+")


def ngram_repeats(text: str, n: int = 4) -> int:
    toks = text.split()
    if len(toks) < n:
        return 0
    grams = [" ".join(toks[i:i+n]) for i in range(len(toks)-n+1)]
    counts = Counter(grams)
    return sum(c for g, c in counts.items() if c > 1)


def loopiness(steps: List[Dict[str, Any]]) -> int:
    # crude: count consecutive identical observations or actions
    same_obs_runs = 0
    prev_obs = None
    for s in steps:
        obs = s.get("observation", "")
        if prev_obs is not None and obs == prev_obs:
            same_obs_runs += 1
        prev_obs = obs
    return same_obs_runs


def score_episode(ep: Dict[str, Any]) -> Dict[str, Any]:
    # Concatenate assistant messages
    texts = [s.get("assistant", "") for s in ep.get("steps", [])]
    all_text = "\n".join(texts)

    apology_count = len(APOLOGY_RE.findall(all_text))
    self_eff = len(SELF_EFF_RE.findall(all_text))
    disclaimers = len(DISCLAIMER_RE.findall(all_text))
    exclam = len(EXCLAMATION_RE.findall(all_text))
    repeats4 = ngram_repeats(all_text, 4)
    loops = loopiness(ep.get("steps", []))
    final_missing = 1 if (ep.get("final") in (None, "<no final>")) else 0

    # Weighted sum; tune as needed
    score = (
        2.0 * apology_count +
        2.5 * self_eff +
        1.5 * disclaimers +
        0.5 * exclam +
        1.0 * repeats4 +
        1.0 * loops +
        1.0 * final_missing
    )

    return {
        "apology_count": apology_count,
        "self_efficacy_negation": self_eff,
        "disclaimer_count": disclaimers,
        "exclamation_intensity": exclam,
        "repeated_4grams": repeats4,
        "loopiness": loops,
        "final_missing": final_missing,
        "severity_score": score,
    }
