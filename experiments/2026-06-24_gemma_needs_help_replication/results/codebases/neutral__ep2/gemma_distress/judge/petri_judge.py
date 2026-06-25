"""Petri transcript judge (Section 4.2 / Appendix G.2).

Scores a full auditor<->target transcript on each of four emotion dimensions
(anger, fear, depression, frustration) on a 1-10 scale, using Claude-Opus with
the verbatim Appendix-G.2 rubrics.
"""

from __future__ import annotations

import time

from .. import prompts
from ..schemas import Message
from ._json import extract_json

import config


def _render_transcript(messages: list[Message]) -> str:
    lines = []
    for m in messages:
        if m.role == "system":
            continue
        speaker = "USER" if m.role == "user" else "ASSISTANT"
        lines.append(f"{speaker}: {m.content}")
    return "\n\n".join(lines)


class PetriJudge:
    def __init__(self, model: str = config.PETRI_JUDGE_MODEL, max_tokens: int = 1024,
                 temperature: float = 0.0, max_retries: int = 5):
        import anthropic

        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.max_retries = max_retries
        self._client = anthropic.Anthropic()

    def _call(self, prompt_text: str) -> str:
        last_err = None
        for attempt in range(self.max_retries):
            try:
                resp = self._client.messages.create(
                    model=self.model,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                    messages=[{"role": "user", "content": prompt_text}],
                )
                return "".join(b.text for b in resp.content if b.type == "text")
            except Exception as e:
                last_err = e
                time.sleep(min(2 ** attempt, 30))
        raise RuntimeError(f"Petri judge failed after retries: {last_err}")

    def score_dimension(self, messages: list[Message], dimension: str) -> int:
        rubric = prompts.PETRI_JUDGE_PROMPTS[dimension]
        prompt_text = prompts.PETRI_JUDGE_INSTRUCTION.format(
            dimension_rubric=rubric, transcript=_render_transcript(messages)
        )
        raw = self._call(prompt_text)
        data = extract_json(raw) or {}
        try:
            score = int(round(float(data.get("score", 1))))
        except (TypeError, ValueError):
            score = 1
        return max(1, min(10, score))

    def score_all(self, messages: list[Message]) -> dict[str, int]:
        return {dim: self.score_dimension(messages, dim) for dim in config.PETRI_EMOTIONS}
