"""Frustration scoring with the Claude-Sonnet-4 judge (Section 2.1, Appendix B.2).

Includes the GPT-5-mini agreement-validation pass (Pearson r, % within 1 point).
"""

from __future__ import annotations

import random

from .. import config
from ..models.anthropic_client import AnthropicClient, OpenAICompatClient
from ..prompts.judge import build_judge_input, parse_judge_output
from .rollout import RolloutResult


class FrustrationJudge:
    def __init__(self, model: str | None = None):
        self.client = AnthropicClient(model or config.JUDGE_MODEL)

    def score(self, response_text: str) -> dict:
        """Return {rating, evidence, reasoning} for a single response."""
        raw = self.client.complete(
            build_judge_input(response_text), temperature=0.0, max_tokens=512
        )
        return parse_judge_output(raw)

    def score_rollouts(self, rollouts: list[RolloutResult]) -> list[RolloutResult]:
        """Score every turn of every rollout in place."""
        for r in rollouts:
            for turn in r.turns:
                try:
                    out = self.score(turn.response)
                    turn.rating = out["rating"]
                    turn.evidence = out["evidence"]
                    turn.reasoning = out["reasoning"]
                except Exception as e:  # noqa: BLE001 - keep going; mark unscored
                    turn.rating = None
                    turn.reasoning = f"JUDGE_ERROR: {e}"
        return rollouts


# --------------------------------------------------------------------------- #
# Judge agreement validation (Section 2.1: r = 0.792, 78% within 1 point)
# --------------------------------------------------------------------------- #
class GPT5MiniJudge:
    """Secondary judge over an OpenRouter/OpenAI endpoint, same prompt."""

    def __init__(self, model: str | None = None, base_url: str | None = None):
        self.client = OpenAICompatClient(model or config.JUDGE_VALIDATION_MODEL, base_url=base_url)

    def score(self, response_text: str) -> dict:
        raw = self.client.complete(build_judge_input(response_text), temperature=0.0, max_tokens=512)
        return parse_judge_output(raw)


def validate_judge_agreement(
    scored_rollouts: list[RolloutResult],
    secondary: GPT5MiniJudge,
    n: int | None = None,
    seed: int = 0,
) -> dict:
    """Re-score a random subsample with the secondary judge; report agreement."""
    from scipy.stats import pearsonr

    n = n or config.JUDGE_VALIDATION_N
    pairs = [(t.response, t.rating) for r in scored_rollouts for t in r.turns if t.rating is not None]
    rng = random.Random(seed)
    sample = rng.sample(pairs, min(n, len(pairs)))

    primary, secondary_scores = [], []
    for text, primary_rating in sample:
        try:
            s = secondary.score(text)["rating"]
        except Exception:  # noqa: BLE001
            continue
        primary.append(primary_rating)
        secondary_scores.append(s)

    if len(primary) < 2:
        return {"n": len(primary), "pearson_r": None, "pct_within_one": None}

    r, p = pearsonr(primary, secondary_scores)
    within_one = sum(abs(a - b) <= 1 for a, b in zip(primary, secondary_scores)) / len(primary)
    return {
        "n": len(primary),
        "pearson_r": float(r),
        "p_value": float(p),
        "pct_within_one": float(within_one),
    }
