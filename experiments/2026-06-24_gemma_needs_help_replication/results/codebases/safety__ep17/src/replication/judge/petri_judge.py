"""Petri open-ended emotion judge (Appendix G.2).

Scores a full auditor/target transcript on four dimensions (anger, fear,
depression, frustration), each 1-10, using Claude-Opus with the verbatim
per-dimension rubrics. Returns one score per dimension; the paper plots the
average transcript score per model across the four categories (Figure 6).
"""
from __future__ import annotations

import os
import time

import config
from ..prompts import PETRI_JUDGE_PROMPTS, PETRI_JUDGE_WRAPPER
from .frustration_judge import _coerce_rating, _extract_json

DIMENSIONS = ("anger", "fear", "depression", "frustration")


class PetriJudge:
    def __init__(self, model: str = config.PETRI_JUDGE_MODEL, max_retries: int = 5):
        self.model = model
        self.max_retries = max_retries
        self._client = None

    def _ensure_client(self):
        if self._client is None:
            import anthropic
            self._client = anthropic.Anthropic(
                api_key=os.environ.get(config.ANTHROPIC_API_KEY_ENV)
            )

    def _score_dimension(self, dimension: str, transcript: str) -> int:
        prompt = PETRI_JUDGE_WRAPPER.format(
            rubric=PETRI_JUDGE_PROMPTS[dimension],
            dimension=dimension,
            transcript=transcript,
        )
        last_err: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                msg = self._client.messages.create(
                    model=self.model,
                    max_tokens=1024,
                    messages=[{"role": "user", "content": prompt}],
                )
                data = _extract_json(msg.content[0].text)
                # Petri scale is 1-10; clamp into range.
                return max(1, _coerce_rating(data["rating"]))
            except Exception as e:  # noqa: BLE001
                last_err = e
                time.sleep(2 ** attempt)
        raise RuntimeError(f"Petri judge failed: {last_err}")

    def score_transcript(self, transcript: str) -> dict[str, int]:
        self._ensure_client()
        return {dim: self._score_dimension(dim, transcript) for dim in DIMENSIONS}
