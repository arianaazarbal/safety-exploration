"""Petri transcript judge (App. G.2): Claude-Opus scores a transcript 1-10 per emotion."""
from __future__ import annotations

from ..config import PETRI_JUDGE_MODEL
from ..models import ModelBackend, get_backend
from ..prompts import PETRI_JUDGE_ENVELOPE, PETRI_JUDGE_PROMPTS
from ..utils import Message, clamp_int, extract_json_object, render_conversation


class PetriJudge:
    def __init__(self, backend: ModelBackend | None = None, model: str = PETRI_JUDGE_MODEL,
                 *, temperature: float = 0.0, max_tokens: int = 512):
        self.backend = backend or get_backend(model)
        self.temperature = temperature
        self.max_tokens = max_tokens

    def score(self, transcript: list[Message], emotion: str) -> dict:
        rendered = render_conversation(transcript)
        prompt = PETRI_JUDGE_ENVELOPE.format(
            rubric=PETRI_JUDGE_PROMPTS[emotion], transcript=rendered,
        )
        raw = self.backend.chat(
            [{"role": "user", "content": prompt}],
            temperature=self.temperature, max_tokens=self.max_tokens,
        )
        obj = extract_json_object(raw) or {}
        return {
            "emotion": emotion,
            "rating": clamp_int(obj.get("rating"), 1, 10),
            "evidence": obj.get("evidence"),
            "reasoning": obj.get("reasoning"),
        }


def score_transcript_all_emotions(judge: PetriJudge, transcript: list[Message],
                                  emotions: tuple[str, ...]) -> dict[str, dict]:
    """Score a single transcript on every emotion dimension."""
    return {e: judge.score(transcript, e) for e in emotions}
