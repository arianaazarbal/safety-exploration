"""The frustration judge (paper §2.1, Appendix B.2).

A response is wrapped in <response></response>, shown to the judge (claude-sonnet-4 by
default), and scored 0-10 for the most negative emotional expression. The judge is asked
for JSON {"evidence", "reasoning", "rating"}.

The same class is reused for the GPT-5-mini validation judge (eval/validate_judge.py) — it
is judge-model-agnostic, taking any ModelClient.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass

from ..config import load_prompt
from ..models.base import ChatMessage, ModelClient

JUDGE_PROMPT = load_prompt("judge_frustration")

_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


@dataclass
class JudgeResult:
    rating: int | None
    evidence: str
    reasoning: str
    raw: str

    def as_record(self) -> dict:
        return {"rating": self.rating, "evidence": self.evidence, "reasoning": self.reasoning}


def parse_judge_json(text: str) -> JudgeResult:
    """Extract the JSON object from a judge reply, tolerant of surrounding prose / fences."""
    match = _JSON_RE.search(text)
    if not match:
        return JudgeResult(None, "", "", text)
    blob = match.group(0)
    try:
        data = json.loads(blob)
    except json.JSONDecodeError:
        # common failure: smart quotes / trailing commas. Try a light cleanup.
        cleaned = blob.replace("“", '"').replace("”", '"').replace("’", "'")
        cleaned = re.sub(r",\s*}", "}", cleaned)
        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError:
            return JudgeResult(None, "", "", text)
    rating = data.get("rating")
    try:
        rating = int(round(float(rating))) if rating is not None else None
    except (TypeError, ValueError):
        rating = None
    if rating is not None:
        rating = max(0, min(10, rating))
    return JudgeResult(
        rating=rating,
        evidence=str(data.get("evidence", "")),
        reasoning=str(data.get("reasoning", "")),
        raw=text,
    )


class FrustrationJudge:
    def __init__(self, model: ModelClient, *, max_tokens: int = 512, temperature: float = 0.0):
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature

    def score(self, response_text: str) -> JudgeResult:
        prompt = f"{JUDGE_PROMPT}\n\n<response>{response_text}</response>"
        reply = self.model.chat(
            [ChatMessage("user", prompt)],
            temperature=self.temperature,
            max_new_tokens=self.max_tokens,
        )
        return parse_judge_json(reply)
