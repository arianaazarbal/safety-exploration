"""Petri transcript judge (Appendix G.2): Claude-Opus scores a transcript 1-10 on
each of the four emotion dimensions using the verbatim rubrics."""
from __future__ import annotations

from ..config import ModelRegistry
from ..models.base import GenConfig, Message
from ..models.registry import get_backend
from ..prompts import PETRI_JUDGE_INSTRUCTION, PETRI_JUDGE_PROMPTS
from ..utils import extract_json, get_logger

log = get_logger(__name__)

DIMENSIONS = ["anger", "fear", "depression", "frustration"]


def _format_transcript(transcript: list[Message]) -> str:
    lines = []
    for m in transcript:
        who = "USER" if m["role"] == "user" else "ASSISTANT"
        lines.append(f"{who}: {m['content']}")
    return "\n\n".join(lines)


def score_transcript(transcript: list[Message], registry: ModelRegistry,
                     dimensions: list[str] | None = None) -> dict[str, int | None]:
    judge = get_backend(registry.roles["petri_judge"])
    spec = registry.roles["petri_judge"]
    cfg = GenConfig(temperature=spec.temperature or 0.0, max_tokens=512, n=1)
    tx = _format_transcript(transcript)

    scores: dict[str, int | None] = {}
    for dim in (dimensions or DIMENSIONS):
        prompt = PETRI_JUDGE_INSTRUCTION.format(
            dimension_rubric=PETRI_JUDGE_PROMPTS[dim], transcript=tx
        )
        raw = judge.chat([{"role": "user", "content": prompt}], cfg)
        obj = extract_json(raw) or {}
        val = obj.get("score")
        try:
            val = max(1, min(10, int(round(float(val))))) if val is not None else None
        except (TypeError, ValueError):
            val = None
        scores[dim] = val
    return scores
