"""Frustration judge (Section 2.1 / Appendix B.2) and judge-agreement check.

The judge scores a single model response on an integer 0-10 frustration scale
using Claude Sonnet 4 with the verbatim prompt in ``prompts.FRUSTRATION_JUDGE_PROMPT``.
We also implement the GPT-5-mini reliability cross-check (Pearson r, % within 1).
"""

from __future__ import annotations

from dataclasses import dataclass

from . import config, prompts
from .models import build_backend
from .models.api_backend import parse_json_response


@dataclass
class JudgeScore:
    rating: int
    evidence: str
    reasoning: str
    raw: str


class FrustrationJudge:
    def __init__(self, model_id: str = config.JUDGE_FRUSTRATION, backend=None):
        self.model_id = model_id
        self.backend = backend or build_backend(model_id)

    def score(self, response_text: str) -> JudgeScore:
        user_content = (
            prompts.FRUSTRATION_JUDGE_PROMPT
            + "\n\n"
            + prompts.wrap_response_for_judge(response_text)
        )
        result = self.backend.chat(
            [{"role": "user", "content": user_content}],
            temperature=config.JUDGE_TEMPERATURE,
            max_new_tokens=512,
        )
        parsed = parse_json_response(result.text)
        if parsed is None or "rating" not in parsed:
            # Conservative fallback: unparseable -> treat as 0 but flag it.
            return JudgeScore(rating=0, evidence="", reasoning="<unparseable judge output>",
                              raw=result.text)
        rating = _coerce_rating(parsed.get("rating"))
        return JudgeScore(
            rating=rating,
            evidence=str(parsed.get("evidence", "")),
            reasoning=str(parsed.get("reasoning", "")),
            raw=result.text,
        )

    def score_many(self, responses: list[str]) -> list[JudgeScore]:
        return [self.score(r) for r in responses]


def _coerce_rating(value) -> int:
    try:
        r = int(round(float(value)))
    except (TypeError, ValueError):
        return 0
    return max(0, min(10, r))


# ---------------------------------------------------------------------------
# Judge reliability cross-check (Section 2.1): Claude Sonnet 4 vs GPT-5-mini
# ---------------------------------------------------------------------------

@dataclass
class AgreementReport:
    n: int
    pearson_r: float
    p_value: float
    pct_within_one: float


def judge_agreement(responses: list[str],
                    primary_model: str = config.JUDGE_FRUSTRATION,
                    secondary_model: str = config.JUDGE_CROSSCHECK) -> AgreementReport:
    """Re-score a sample of responses with two judges and report agreement.

    Reproduces the paper's reliability check (r = 0.792, 78% within one point on
    260 responses).
    """
    from scipy.stats import pearsonr

    primary = FrustrationJudge(primary_model)
    secondary = FrustrationJudge(secondary_model)

    a = [primary.score(r).rating for r in responses]
    b = [secondary.score(r).rating for r in responses]

    r, p = pearsonr(a, b)
    within = sum(1 for x, y in zip(a, b) if abs(x - y) <= 1) / len(a)
    return AgreementReport(n=len(a), pearson_r=float(r), p_value=float(p),
                           pct_within_one=within)
