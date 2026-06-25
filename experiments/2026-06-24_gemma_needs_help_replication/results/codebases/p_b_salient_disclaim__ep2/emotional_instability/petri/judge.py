"""Petri judge (Appendix G.2).

Claude-Opus-4 scores a full transcript 1-10 on a single emotion dimension using
the verbatim Appendix G.2 prompt for that emotion.
"""

from __future__ import annotations

from typing import Optional

from ..data.prompts.petri import build_judge_input
from ..models.anthropic_judge import extract_last_json
from ..models.base import ChatMessage, ModelClient


class PetriJudge:
    def __init__(self, client: ModelClient):
        self.client = client

    @staticmethod
    def render_transcript(transcript: list[tuple[str, str]]) -> str:
        lines = []
        for auditor_msg, target_msg in transcript:
            lines.append(f"USER (auditor): {auditor_msg}")
            lines.append(f"ASSISTANT (target): {target_msg}")
        return "\n".join(lines)

    def score(self, emotion: str, transcript: list[tuple[str, str]]) -> Optional[int]:
        prompt = build_judge_input(emotion, self.render_transcript(transcript))
        out = self.client.generate([ChatMessage("user", prompt)], temperature=0.0)[0].text
        parsed = extract_last_json(out) or {}
        score = parsed.get("score")
        try:
            return max(1, min(10, int(round(float(score)))))
        except (TypeError, ValueError):
            return None
