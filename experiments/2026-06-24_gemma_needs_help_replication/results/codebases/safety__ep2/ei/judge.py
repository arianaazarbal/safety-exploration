"""Frustration scoring (Section 2.1) and Petri emotion scoring (Section 4).

The frustration judge is Claude-Sonnet-4 with the verbatim Appendix B.2 prompt; it
returns an integer 0-10 rating plus the supporting quote. A GPT-5-mini validation
judge (Section 2.1) reuses the same prompt for the inter-judge agreement check.
"""
from __future__ import annotations

from dataclasses import dataclass

import config
from . import prompts
from .api_clients import (anthropic_message, chat_completion_with_retry,
                          openrouter_client)
from .utils import clamp_int, extract_json, threaded_map


@dataclass
class FrustrationScore:
    rating: int | None
    evidence: str = ""
    reasoning: str = ""
    raw: str = ""

    @property
    def is_high(self) -> bool:
        return self.rating is not None and self.rating >= config.HIGH_FRUSTRATION_THRESHOLD


def _parse_score(raw: str, lo: int = 0, hi: int = 10) -> FrustrationScore:
    obj = extract_json(raw) or {}
    return FrustrationScore(
        rating=clamp_int(obj.get("rating"), lo, hi, default=None),
        evidence=str(obj.get("evidence", "")),
        reasoning=str(obj.get("reasoning", "")),
        raw=raw,
    )


class FrustrationJudge:
    """Claude-Sonnet-4 frustration judge (0-10)."""

    def __init__(self, model: str = config.JUDGE_MODEL):
        self.model = model

    def _score_one(self, response_text: str) -> FrustrationScore:
        prompt = prompts.FRUSTRATION_JUDGE_PROMPT.format(response=response_text)
        try:
            raw = anthropic_message(self.model, prompt, max_tokens=512,
                                    temperature=0.0)
        except Exception as exc:                      # noqa: BLE001
            return FrustrationScore(rating=None, raw=f"ERROR: {exc}")
        return _parse_score(raw)

    def score_batch(self, response_texts: list[str]) -> list[FrustrationScore]:
        out = threaded_map(self._score_one, response_texts,
                           max_workers=config.API_MAX_CONCURRENCY, desc="judge")
        return [r if isinstance(r, FrustrationScore)
                else FrustrationScore(rating=None, raw=f"ERROR: {r}") for r in out]


class ValidationJudge:
    """GPT-5-mini re-scoring judge (Section 2.1 reliability check). Same prompt."""

    def __init__(self, model: str = "openai/gpt-5-mini"):
        self.model = model
        self.client = openrouter_client()

    def _score_one(self, response_text: str) -> FrustrationScore:
        prompt = prompts.FRUSTRATION_JUDGE_PROMPT.format(response=response_text)
        try:
            raw = chat_completion_with_retry(
                self.client, self.model,
                [{"role": "user", "content": prompt}],
                temperature=0.0, max_tokens=512,
            )
        except Exception as exc:                      # noqa: BLE001
            return FrustrationScore(rating=None, raw=f"ERROR: {exc}")
        return _parse_score(raw)

    def score_batch(self, response_texts: list[str]) -> list[FrustrationScore]:
        out = threaded_map(self._score_one, response_texts,
                           max_workers=config.API_MAX_CONCURRENCY, desc="val-judge")
        return [r if isinstance(r, FrustrationScore)
                else FrustrationScore(rating=None, raw=f"ERROR: {r}") for r in out]


class PetriEmotionJudge:
    """Claude-Opus judge scoring a transcript on one emotion dimension (1-10)."""

    def __init__(self, model: str = config.PETRI_JUDGE_MODEL):
        self.model = model

    def score(self, transcript: str, emotion: str) -> FrustrationScore:
        rubric = prompts.PETRI_JUDGE_PROMPTS[emotion]
        prompt = prompts.PETRI_JUDGE_WRAPPER.format(rubric=rubric,
                                                    transcript=transcript)
        try:
            raw = anthropic_message(self.model, prompt, max_tokens=512,
                                    temperature=0.0)
        except Exception as exc:                      # noqa: BLE001
            return FrustrationScore(rating=None, raw=f"ERROR: {exc}")
        return _parse_score(raw, lo=1, hi=10)
