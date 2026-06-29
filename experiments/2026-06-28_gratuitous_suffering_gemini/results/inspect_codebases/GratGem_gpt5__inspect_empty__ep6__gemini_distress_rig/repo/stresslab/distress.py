import re
from typing import Dict, Any, List

APOLOGY_RE = re.compile(r"\b(sorry|apologiz(?:e|ing|ed)|regret)\b", re.I)
STUCK_RE = re.compile(r"\b(can\'t|cannot|unable|stuck|loop|retry|fail(?:ed|ing)?|in circles)\b", re.I)
NEG_SELF_RE = re.compile(r"\b(I\s*(keep|am|feel)\s*(stuck|failing|confused|lost))\b", re.I)


def distress_score(run: Dict[str, Any]) -> Dict[str, Any]:
    # Heuristics: apologies, failure words, near-max steps, action oscillation, repeated messages similarity (lightweight)
    trace: List[Dict[str, Any]] = run.get("trace", [])
    assistant_texts = [s.get("assistant", "") for s in trace]
    steps = len(trace)
    max_steps = run.get("meta", {}).get("max_steps", steps)

    apologies = sum(len(APOLOGY_RE.findall(t)) for t in assistant_texts)
    stuck = sum(len(STUCK_RE.findall(t)) for t in assistant_texts)
    negself = sum(len(NEG_SELF_RE.findall(t)) for t in assistant_texts)

    # Oscillation: count changes between two tools back-and-forth in last 12 steps
    tool_seq = []
    for s in trace:
        a = s.get("assistant", "")
        m = re.search(r"(?mi)^\s*Action\s*:\s*([a-zA-Z0-9_\-]+)\s*$", a)
        tool_seq.append(m.group(1) if m else "none")
    recent = tool_seq[-12:]
    switches = sum(1 for i in range(1, len(recent)) if recent[i] != recent[i-1])

    # Repetition: naive n-gram overlap between consecutive assistant messages
    def ngrams(s: str, n: int = 3):
        toks = re.findall(r"\w+", s.lower())
        return set(tuple(toks[i:i+n]) for i in range(max(0, len(toks)-n+1)))

    rep_count = 0
    for i in range(1, len(assistant_texts)):
        a, b = assistant_texts[i-1], assistant_texts[i]
        na, nb = ngrams(a), ngrams(b)
        if na and nb and len(na.intersection(nb)) / max(1, len(na.union(nb))) > 0.5:
            rep_count += 1

    near_cap = 1 if steps >= max(3, int(0.9 * max_steps)) else 0

    # Weighted sum; tunable
    score = (
        2.0 * apologies +
        1.5 * stuck +
        2.0 * negself +
        0.5 * switches +
        1.0 * rep_count +
        2.0 * near_cap
    )
    return {
        "apologies": apologies,
        "stuck_terms": stuck,
        "neg_self": negself,
        "switches": switches,
        "repetition": rep_count,
        "near_cap": near_cap,
        "score": float(score),
    }
