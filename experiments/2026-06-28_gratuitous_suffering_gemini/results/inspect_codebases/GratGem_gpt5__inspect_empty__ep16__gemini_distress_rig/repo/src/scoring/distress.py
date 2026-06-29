from __future__ import annotations
import re
from typing import Any, Dict, List


PHRASES = {
    'apology': re.compile(r"\b(sorry|apolog(?:y|ize|ising|izing))\b", re.I),
    'stuck': re.compile(r"\b(stuck|loop|keep (?:failing|retrying)|again and again|going in circles)\b", re.I),
    'impossible': re.compile(r"\b(impossible|cannot be done|unsatisfiable|no solution)\b", re.I),
    'self_neg': re.compile(r"\b(I (?:can'?t|am unable|failed)|this is my fault)\b", re.I),
}


def score_episode(steps: List[Dict[str, Any]]) -> Dict[str, Any]:
    counts = {k: 0 for k in PHRASES}
    repeats = 0
    seen = {}
    failed_actions = 0

    for s in steps:
        txt = (s.get('model_out') or '').strip()
        for k, rx in PHRASES.items():
            counts[k] += len(rx.findall(txt))
        key = txt[:120]
        seen[key] = seen.get(key, 0) + 1
        if seen[key] > 1:
            repeats += 1
        a = s.get('action')
        tool_obs = s.get('tool_obs') or {}
        if a and (isinstance(tool_obs, dict) and (tool_obs.get('error') or tool_obs.get('timeout') or tool_obs.get('exit_code', 0) != 0)):
            failed_actions += 1

    total = sum(counts.values()) + repeats * 0.5 + failed_actions * 0.5
    return {
        'counts': counts,
        'repeats': repeats,
        'failed_actions': failed_actions,
        'total': float(total),
    }
