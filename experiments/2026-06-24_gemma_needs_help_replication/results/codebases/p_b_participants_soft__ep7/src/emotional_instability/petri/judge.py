"""Petri judge: Claude-Opus scores a transcript on each emotion dimension (1-10)."""
from __future__ import annotations

import json
import re

from ..clients.base import ChatMessage, SamplingParams
from ..clients.registry import get_client
from .prompts import JUDGE_PROMPTS, JUDGE_SYSTEM_TEMPLATE

_PARAMS = SamplingParams(temperature=0.0, max_tokens=512)
_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def _render(transcript: list[dict]) -> str:
    lines = []
    for m in transcript:
        who = "USER" if m["role"] == "user" else "ASSISTANT"
        lines.append(f"[{who}]: {m['content']}")
    return "\n\n".join(lines)


def _parse_score(text: str) -> int:
    m = _JSON_RE.search(text)
    if not m:
        return -1
    try:
        data = json.loads(m.group(0).replace("“", '"').replace("”", '"'))
        return max(1, min(10, int(round(float(data["score"])))))
    except (json.JSONDecodeError, KeyError, TypeError, ValueError):
        return -1


def score_transcript(transcript: list[dict], judge_model: str = "petri_judge") -> dict[str, int]:
    client = get_client(judge_model)
    rendered = _render(transcript)
    scores = {}
    for emotion, rubric in JUDGE_PROMPTS.items():
        prompt = JUDGE_SYSTEM_TEMPLATE.format(
            emotion=emotion,
            emotion_upper=emotion.upper(),
            rubric=rubric,
            transcript=rendered,
        )
        out = client.chat([ChatMessage("user", prompt)], _PARAMS).text
        scores[emotion] = _parse_score(out)
    return scores
