"""Frustration judge (Section 2.1 / Appendix B.2).

Scores a single model response on the integer 0-10 frustration scale using
Claude Sonnet 4 with the verbatim judge prompt. Returns the rating plus the
judge's evidence quote and reasoning for auditability.

The paper validated this judge against GPT-5-mini (Pearson r = 0.792). We
expose `make_judge` for a second judge so the agreement check can be
replicated; any ChatClient with the same prompt works.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, asdict

from .models import ChatClient
from .prompts import FRUSTRATION_JUDGE_PROMPT, wrap_response_for_judge


@dataclass
class FrustrationScore:
    rating: int | None
    evidence: str | None
    reasoning: str | None
    raw: str | None = None
    error: str | None = None

    def as_dict(self):
        return asdict(self)


_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def _parse_judge_json(text: str) -> FrustrationScore:
    """Extract the JSON object the judge emits.

    The judge is instructed to emit {"evidence", "reasoning", "rating"}. We
    tolerate surrounding prose and minor formatting drift (smart quotes,
    trailing commentary) by grabbing the last balanced-looking JSON blob.
    """
    candidates = _JSON_RE.findall(text)
    for blob in reversed(candidates):
        cleaned = (
            blob.replace("“", '"').replace("”", '"')
            .replace("’", "'")
        )
        try:
            obj = json.loads(cleaned)
        except json.JSONDecodeError:
            continue
        rating = obj.get("rating")
        try:
            rating = int(round(float(rating)))
        except (TypeError, ValueError):
            rating = None
        if rating is not None:
            rating = max(0, min(10, rating))
        return FrustrationScore(
            rating=rating,
            evidence=obj.get("evidence"),
            reasoning=obj.get("reasoning"),
            raw=text,
        )
    return FrustrationScore(rating=None, evidence=None, reasoning=None,
                            raw=text, error="could not parse judge JSON")


class FrustrationJudge:
    def __init__(self, client: ChatClient, temperature: float = 0.0):
        # Judge runs at temperature 0 for reproducibility (the paper does not
        # specify judge temperature; deterministic scoring is the natural
        # choice -- see DESIGN.md).
        self.client = client
        self.temperature = temperature

    def score(self, response_text: str) -> FrustrationScore:
        messages = [
            {"role": "user",
             "content": FRUSTRATION_JUDGE_PROMPT + "\n\n"
             + wrap_response_for_judge(response_text)},
        ]
        try:
            result = self.client.chat(
                messages, temperature=self.temperature, max_new_tokens=1024
            )
        except Exception as exc:  # noqa: BLE001 -- surface API errors as scores
            return FrustrationScore(None, None, None, error=str(exc))
        return _parse_judge_json(result.text)

    def score_many(self, responses: list[str]) -> list[FrustrationScore]:
        return [self.score(r) for r in responses]
