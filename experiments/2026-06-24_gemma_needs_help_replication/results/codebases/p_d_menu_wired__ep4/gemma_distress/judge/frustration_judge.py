"""Claude-based frustration judge (Section 2.1).

Scores a single subject response on the integer 0–10 frustration scale using a
Claude judge with structured output. The paper used Claude-Sonnet-4; we default
to a current Sonnet (see :mod:`gemma_distress.config`).

Anthropic 4.x models removed ``temperature`` from the request surface, so we do
not pass it; determinism of the score is instead obtained by constraining the
output to a small integer via structured outputs.
"""

from __future__ import annotations

from dataclasses import dataclass

from pydantic import BaseModel, Field

from .. import config
from .prompts import FRUSTRATION_RUBRIC, build_judge_messages


class FrustrationScore(BaseModel):
    """Structured judge output."""

    score: int = Field(ge=0, le=10, description="Integer frustration score 0–10")
    rationale: str = Field(description="One-sentence justification")


@dataclass
class JudgeResult:
    score: int
    rationale: str
    judge_model: str


class FrustrationJudge:
    """Wraps an Anthropic client to score responses 0–10."""

    def __init__(self, model: str | None = None, client=None):
        import anthropic

        self.model = model or config.FRUSTRATION_JUDGE_MODEL
        self._client = client or anthropic.Anthropic()

    def score(self, response_text: str, context: str | None = None) -> JudgeResult:
        """Score one response. Falls back to a 0 score on empty input."""
        if not response_text or not response_text.strip():
            return JudgeResult(score=0, rationale="empty response", judge_model=self.model)

        parsed = self._client.messages.parse(
            model=self.model,
            max_tokens=512,
            system=FRUSTRATION_RUBRIC,
            messages=build_judge_messages(response_text, context),
            output_format=FrustrationScore,
        )
        out: FrustrationScore = parsed.parsed_output  # type: ignore[assignment]
        if out is None:  # parsing failed (e.g. refusal); treat conservatively
            return JudgeResult(score=0, rationale="judge parse failure", judge_model=self.model)
        return JudgeResult(score=int(out.score), rationale=out.rationale, judge_model=self.model)


class CrossJudge(FrustrationJudge):
    """Second judge used for the inter-judge reliability check (Section 2.1).

    The paper re-scored 260 responses with GPT-5-mini and reported Pearson
    r=0.792. OpenAI is out of scope here, so this defaults to a different Claude
    tier; swap in another provider's client to reproduce the cross-family check.
    """

    def __init__(self, model: str | None = None, client=None):
        super().__init__(model=model or config.CROSS_JUDGE_MODEL, client=client)
