"""Petri judge: Claude-Opus scores a transcript 1-10 on a single emotion
dimension using the Appendix G.2 rubric."""

from __future__ import annotations

import json
import re

from src.models import load_model
from src.models.base import Message
from src.prompts.petri_prompts import JUDGE_OUTPUT_INSTRUCTION, JUDGE_PROMPTS

_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


class PetriJudge:
    def __init__(self, spec):
        self.model = load_model(spec)

    def score(self, emotion: str, transcript_text: str) -> int:
        system = JUDGE_PROMPTS[emotion]
        user = JUDGE_OUTPUT_INSTRUCTION.format(transcript=transcript_text)
        raw = self.model.generate(
            [Message("system", system), Message("user", user)],
            temperature=0.0,
            max_new_tokens=512,
        )
        m = _JSON_RE.search(raw)
        if m:
            blob = m.group(0).replace("“", '"').replace("”", '"').replace("’", "'")
            try:
                return max(1, min(10, int(round(float(json.loads(blob)["score"])))))
            except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                pass
        m2 = re.search(r'"?score"?\s*[:=]\s*(\d+)', raw)
        return max(1, min(10, int(m2.group(1)))) if m2 else -1
