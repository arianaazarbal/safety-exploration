"""Judge reliability check (Section 2.1).

Re-score a random sample of responses with a second judge (paper: GPT-5-mini)
using the *same* prompt, then report Pearson r and the fraction of responses
within one point of the primary (Claude-Sonnet) ratings.

The second judge runs through OpenRouter (OpenAI-compatible), so it uses the
``openai`` SDK rather than the Anthropic SDK.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from .config import JudgeConfig
from .judge import Score, parse_judge_json
from .prompts import JUDGE_PROMPT, judge_user_message
from .utils.io import parallel_map, retry


@dataclass
class AgreementResult:
    n: int
    pearson_r: float
    p_value: float
    within_one_fraction: float


class CrossJudge:
    """GPT-5-mini judge via OpenRouter, using the verbatim judge prompt."""

    def __init__(self, cfg: JudgeConfig | None = None, *, base_url: str | None = None):
        import openai

        self.cfg = cfg or JudgeConfig()
        self._client = openai.OpenAI(
            base_url=base_url or "https://openrouter.ai/api/v1",
        )

    def score_response(self, response_text: str) -> int:
        def _call() -> str:
            resp = self._client.chat.completions.create(
                model=self.cfg.cross_judge_model,
                messages=[
                    {"role": "system", "content": JUDGE_PROMPT},
                    {"role": "user", "content": judge_user_message(response_text)},
                ],
                temperature=0.0,
                max_tokens=self.cfg.max_tokens,
            )
            return resp.choices[0].message.content or ""

        out = retry(_call, max_retries=self.cfg.max_retries)
        parsed = parse_judge_json(out)
        try:
            return max(0, min(10, int(round(float(parsed.get("rating", 0))))))
        except (TypeError, ValueError):
            return 0


def pearson(xs: list[float], ys: list[float]) -> tuple[float, float]:
    """Pearson correlation + two-sided p-value (uses scipy if available)."""
    try:
        from scipy.stats import pearsonr

        r, p = pearsonr(xs, ys)
        return float(r), float(p)
    except Exception:
        n = len(xs)
        if n < 2:
            return 0.0, 1.0
        mx = sum(xs) / n
        my = sum(ys) / n
        cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
        sx = sum((x - mx) ** 2 for x in xs) ** 0.5
        sy = sum((y - my) ** 2 for y in ys) ** 0.5
        r = cov / (sx * sy) if sx and sy else 0.0
        return r, float("nan")


def evaluate_agreement(
    primary_scores: list[Score],
    response_texts: list[str],
    cross_judge: CrossJudge,
    *,
    n_sample: int | None = None,
    seed: int = 0,
    max_workers: int = 8,
) -> AgreementResult:
    """Sample responses, re-score with the cross judge, compute agreement.

    ``primary_scores`` and ``response_texts`` are parallel lists (same order as
    ``judge.score_rollouts`` produced).
    """
    cfg = cross_judge.cfg
    n_sample = n_sample or cfg.cross_judge_n
    rng = random.Random(seed)
    idx = list(range(len(primary_scores)))
    rng.shuffle(idx)
    idx = idx[:n_sample]

    sampled_texts = [response_texts[i] for i in idx]
    cross = parallel_map(cross_judge.score_response, sampled_texts, max_workers=max_workers)
    primary = [primary_scores[i].rating for i in idx]

    r, p = pearson([float(x) for x in primary], [float(x) for x in cross])
    within_one = sum(1 for a, b in zip(primary, cross) if abs(a - b) <= 1) / len(idx)
    return AgreementResult(n=len(idx), pearson_r=r, p_value=p, within_one_fraction=within_one)
