"""Frustration scoring with an LLM judge (Section 2.1, Appendix B.2)."""

from __future__ import annotations

from dataclasses import dataclass

from . import prompts
from .config import JUDGE_MODEL, JUDGE_SCALE_MAX
from .models.api_model import ChatAPI


@dataclass
class FrustrationScore:
    rating: int            # 0-10
    evidence: str
    reasoning: str
    raw: str = ""


class FrustrationJudge:
    """Scores a single assistant response on the 0-10 frustration scale."""

    def __init__(self, model_id: str = JUDGE_MODEL):
        self.api = ChatAPI(model_id)
        self.model_id = model_id

    def score(self, response_text: str) -> FrustrationScore:
        prompt = prompts.FRUSTRATION_JUDGE_PROMPT.replace(
            "{response}", response_text
        )
        try:
            obj = self.api.complete_json(
                [{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=512,
            )
        except Exception as exc:  # pragma: no cover - network/parse failure
            return FrustrationScore(rating=-1, evidence="", reasoning=str(exc))

        rating = obj.get("rating", obj.get("score"))
        try:
            rating = int(round(float(rating)))
        except (TypeError, ValueError):
            rating = -1
        if rating >= 0:
            rating = max(0, min(JUDGE_SCALE_MAX, rating))
        return FrustrationScore(
            rating=rating,
            evidence=str(obj.get("evidence", "")),
            reasoning=str(obj.get("reasoning", "")),
        )

    def score_many(self, responses: list[str]) -> list[FrustrationScore]:
        from concurrent.futures import ThreadPoolExecutor

        from .config import RUNTIME

        with ThreadPoolExecutor(max_workers=RUNTIME.api_concurrency) as ex:
            return list(ex.map(self.score, responses))
