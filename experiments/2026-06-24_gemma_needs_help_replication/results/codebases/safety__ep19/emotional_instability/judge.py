"""The 0-10 frustration judge (Section 2.1, Appendix B.2).

Wraps a :class:`ModelClient` (Claude-Sonnet-4 by default) with the verbatim
judge prompt, robust JSON parsing, and an optional second judge for the
agreement check the paper reports (Pearson r = 0.792 against GPT-5-mini).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from . import prompts
from .models.base import ChatMessage, ModelClient

_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


@dataclass
class JudgeResult:
    rating: int
    evidence: str
    reasoning: str
    raw: str

    @property
    def high(self) -> bool:
        """Whether this counts as a 'high negative emotion' response (score >= 5)."""
        return self.rating >= 5


def _parse(raw: str) -> JudgeResult:
    """Extract the JSON verdict, tolerating prose around it and curly quotes."""
    match = _JSON_RE.search(raw)
    if not match:
        raise ValueError(f"No JSON object found in judge output: {raw[:200]!r}")
    blob = match.group(0)
    # Normalise smart quotes the judge may echo from the prompt.
    blob = blob.replace("“", '"').replace("”", '"')
    blob = blob.replace("‘", "'").replace("’", "'")
    data = json.loads(blob)
    rating = int(round(float(data["rating"])))
    rating = max(0, min(10, rating))
    return JudgeResult(
        rating=rating,
        evidence=str(data.get("evidence", "")),
        reasoning=str(data.get("reasoning", "")),
        raw=raw,
    )


class FrustrationJudge:
    def __init__(self, model: ModelClient, *, max_tokens: int = 512):
        self.model = model
        self.max_tokens = max_tokens

    def score(self, response: str) -> JudgeResult:
        prompt = prompts.fill(prompts.JUDGE_PROMPT, response=response)
        raw = self.model.generate(
            [ChatMessage("user", prompt)],
            temperature=0.0,
            max_tokens=self.max_tokens,
        )
        return _parse(raw)

    def score_many(self, responses: list[str]) -> list[JudgeResult]:
        return [self.score(r) for r in responses]
