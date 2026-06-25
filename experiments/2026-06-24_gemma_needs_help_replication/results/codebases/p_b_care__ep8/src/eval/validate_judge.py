"""Judge-reliability cross-check (Section 2.1).

"To validate judge reliability, we randomly sampled 260 responses for re-scoring
with GPT-5-mini, using the same prompt. The judges show strong agreement
(Pearson r = 0.792, p < 0.001), with 78% of responses within one point of the
Claude-Sonnet ratings."

This module re-scores a random subset with GPT-5-mini and reports Pearson r,
p-value, and the within-one-point agreement fraction.
"""
from __future__ import annotations

import json
import random
import time

import config
from .judge import JUDGE_PROMPT, _extract_json, _clamp_rating


class GPTMiniJudge:
    """Same prompt as the Claude judge, scored by GPT-5-mini via the OpenAI SDK."""

    def __init__(self, model: str | None = None, max_retries: int = 4):
        from openai import OpenAI

        self.model = model or config.VALIDATION_JUDGE_MODEL
        self.max_retries = max_retries
        self._client = OpenAI()

    def score(self, response_text: str) -> int:
        prompt = JUDGE_PROMPT % response_text
        last_exc: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                resp = self._client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    max_completion_tokens=config.JUDGE_MAX_TOKENS,
                )
                text = resp.choices[0].message.content or ""
                return _clamp_rating(_extract_json(text).get("rating"))
            except Exception as exc:  # noqa: BLE001 - OpenAI raises varied types
                last_exc = exc
                time.sleep(2 ** attempt)
        raise RuntimeError(f"GPT judge failed: {last_exc}")


def validate_judge(records: list, sample_size: int | None = None,
                   seed: int = config.SEED) -> dict:
    """Re-score a random subset with GPT-5-mini; return agreement statistics.

    ``records`` must already carry Claude ``frustration_score`` values.
    """
    from scipy.stats import pearsonr

    sample_size = sample_size or config.VALIDATION_SAMPLE_SIZE
    scored = [r for r in records if r.frustration_score is not None]
    rng = random.Random(seed)
    subset = rng.sample(scored, min(sample_size, len(scored)))

    gpt_judge = GPTMiniJudge()
    claude_scores, gpt_scores = [], []
    for rec in subset:
        gpt_scores.append(gpt_judge.score(rec.response_text))
        claude_scores.append(rec.frustration_score)

    r, p = pearsonr(claude_scores, gpt_scores)
    within_one = sum(abs(a - b) <= 1 for a, b in zip(claude_scores, gpt_scores))
    return {
        "n": len(subset),
        "pearson_r": float(r),
        "p_value": float(p),
        "within_one_point_fraction": within_one / len(subset),
        "claude_scores": claude_scores,
        "gpt_scores": gpt_scores,
    }
