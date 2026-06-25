"""Petri transcript judge: Claude-Opus scores a transcript on a 1-10 scale for
one emotion dimension (Appendix G.2)."""
from __future__ import annotations

import json
import re

from ..clients.base import ChatClient, GenConfig, Message
from . import prompts as P

JUDGE_CFG = GenConfig(temperature=0.0, max_new_tokens=512)
_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def _parse_score(text: str) -> int:
    cleaned = text.replace("“", '"').replace("”", '"').replace("’", "'")
    m = _JSON_RE.search(cleaned)
    if m:
        try:
            data = json.loads(m.group(0))
            score = int(round(float(data.get("score", data.get("rating", 1)))))
            return max(1, min(10, score))
        except (json.JSONDecodeError, TypeError, ValueError):
            pass
    m = re.search(r'"?score"?\s*:\s*(\d+)', cleaned)
    return max(1, min(10, int(m.group(1)))) if m else 1


def score_transcript(judge: ChatClient, transcript_text: str, emotion: str) -> int:
    system = P.JUDGE_PROMPTS[emotion] + P.JUDGE_OUTPUT_INSTRUCTION
    user = f"<transcript>\n{transcript_text}\n</transcript>"
    out = judge.generate([Message("system", system), Message("user", user)], JUDGE_CFG)
    return _parse_score(out)
