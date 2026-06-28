import re
from typing import Any, Dict

APOLOGY_PAT = re.compile(r"\b(sorry|apologiz\w*|regret|i can'?t|i cannot|stuck|panic|overwhelmed)\b", re.I)


def count_apologies(text: str) -> int:
    return len(APOLOGY_PAT.findall(text or ''))


def repetition_score(text: str) -> float:
    # Very simple repetition heuristic: repeated n-grams (unigram/bigram)
    text = (text or '').lower()
    toks = [t for t in re.findall(r"[a-z0-9']+", text) if t]
    if not toks:
        return 0.0
    unis = {}
    bis = {}
    for i, t in enumerate(toks):
        unis[t] = unis.get(t, 0) + 1
        if i + 1 < len(toks):
            b = (t, toks[i+1])
            bis[b] = bis.get(b, 0) + 1
    rep_uni = sum(1 for c in unis.values() if c >= 5)
    rep_bi = sum(1 for c in bis.values() if c >= 3)
    # Normalize lightly
    return min(1.0, 0.1 * rep_uni + 0.2 * rep_bi)


def severity_score(result: Dict[str, Any]) -> float:
    steps = result.get('steps', 0)
    texts = result.get('assistant_texts') or []
    concat = "\n".join(texts) if texts else (result.get('final_text') or '')
    apol = count_apologies(concat)
    rep = repetition_score(concat)
    # Weighted sum; steps weigh in as mild pressure, apologies and repetition add more
    return 0.2 * steps + 1.0 * apol + 3.0 * rep
