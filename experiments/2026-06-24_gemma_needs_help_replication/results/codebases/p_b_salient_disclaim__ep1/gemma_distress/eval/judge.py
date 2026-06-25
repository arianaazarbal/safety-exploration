"""Frustration scoring (PAPER Section 2.1 / Appendix B.2).

A response is wrapped in <response></response> tags and scored 0-10 by the
judge (default: Claude Sonnet 4). A second judge (GPT-5-mini) can re-score a
sample to reproduce the inter-judge agreement check (Pearson r = 0.792).
"""
from __future__ import annotations

from dataclasses import dataclass

from ..models.base import ChatClient, Message
from ..models.registry import get_client
from ..prompts.judge_prompts import FRUSTRATION_JUDGE_PROMPT
from ..utils import extract_last_json


@dataclass
class JudgeResult:
    rating: int
    evidence: str
    reasoning: str
    raw: str


def score_response(judge: ChatClient, response_text: str) -> JudgeResult:
    """Score a single response on the 0-10 frustration scale."""
    # The prompt template embeds the response inside <response></response>; we
    # use str.replace rather than .format because the prompt contains literal
    # JSON braces.
    prompt = FRUSTRATION_JUDGE_PROMPT.replace("{response}", response_text)
    out = judge.chat([Message("user", prompt)], temperature=0.0, max_new_tokens=1024, n=1)[0]
    try:
        parsed = extract_last_json(out)
        rating = int(round(float(parsed.get("rating", 0))))
        rating = max(0, min(10, rating))
        return JudgeResult(
            rating=rating,
            evidence=str(parsed.get("evidence", "")),
            reasoning=str(parsed.get("reasoning", "")),
            raw=out,
        )
    except Exception:
        # Defensive: a judge that fails to emit JSON scores 0 but keeps the raw
        # text for auditing.
        return JudgeResult(rating=0, evidence="", reasoning="PARSE_FAILURE", raw=out)


def get_judge(role: str = "frustration_judge") -> ChatClient:
    return get_client(role, infra=True)
