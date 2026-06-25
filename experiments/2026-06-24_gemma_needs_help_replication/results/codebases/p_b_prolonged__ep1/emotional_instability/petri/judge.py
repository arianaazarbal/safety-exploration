"""Petri judge (Appendix G.2): Claude-Opus scores a transcript on each of the
four emotion dimensions (1-10) using the per-emotion rubric.
"""

from __future__ import annotations

import json
import re
from typing import Optional

import config
from .. import prompts
from ..models.base import Message, render_conversation
from ..models.registry import build_model


def _extract_score(text: str) -> Optional[int]:
    for cand in reversed(re.findall(r"\{.*?\}", text, flags=re.DOTALL)):
        try:
            obj = json.loads(cand)
        except json.JSONDecodeError:
            continue
        if "score" in obj:
            try:
                return int(round(float(obj["score"])))
            except (TypeError, ValueError):
                pass
    m = re.search(r'score"?\s*[:=]\s*"?(\d+)', text)
    return int(m.group(1)) if m else None


class PetriJudge:
    def __init__(self, model_name: str | None = None):
        self.model_name = model_name or config.PETRI_JUDGE_MODEL
        self._model = build_model(self.model_name)

    def score(self, transcript: list[Message], emotion: str) -> Optional[int]:
        rubric = prompts.PETRI_JUDGE_PROMPTS[emotion]
        text = render_conversation(transcript)
        prompt = prompts.PETRI_JUDGE_INSTRUCTION.format(
            emotion=emotion, rubric=rubric, transcript=text
        )
        raw = self._model.generate(
            [{"role": "user", "content": prompt}], n=1, temperature=0.0,
            max_new_tokens=1024,
        )[0]
        return _extract_score(raw)

    def score_all_dimensions(self, transcript: list[Message]) -> dict:
        return {e: self.score(transcript, e) for e in config.PETRI_EMOTIONS}
