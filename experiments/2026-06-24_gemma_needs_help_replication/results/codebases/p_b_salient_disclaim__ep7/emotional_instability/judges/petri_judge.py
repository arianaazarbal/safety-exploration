"""Petri transcript scoring (Appendix G.2).

Claude-Opus scores a full auditor<->target transcript on a 1-10 scale for one
emotion dimension. The four dimensions (anger, fear, depression, frustration)
are each scored with their own rubric prompt.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import config
from ..prompts import build_petri_judge_user_message
from ._llm import anthropic_complete, extract_json


@dataclass
class PetriScore:
    emotion: str
    score: Optional[int]              # 1-10
    reasoning: Optional[str] = None
    raw: Optional[str] = None


def score_petri_transcript(emotion: str, transcript_text: str, *,
                           model: Optional[str] = None) -> PetriScore:
    model = model or config.PETRI_JUDGE_MODEL
    prompt = build_petri_judge_user_message(emotion, transcript_text)
    raw = anthropic_complete(model, prompt, max_tokens=512, temperature=0.0)
    obj = extract_json(raw) or {}
    score = obj.get("score")
    try:
        score = int(round(float(score)))
        score = max(1, min(10, score))
    except (TypeError, ValueError):
        score = None
    return PetriScore(emotion=emotion, score=score,
                      reasoning=obj.get("reasoning"), raw=raw)
