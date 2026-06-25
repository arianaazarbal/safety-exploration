"""Judge-reliability validation (Section 2.1).

The paper re-scores 260 randomly-sampled responses with GPT-5-mini and reports
Pearson r and the fraction within one point of the Claude-Sonnet ratings
(target: r = 0.792, 78% within one point).

This module re-scores a random sample with the agreement judge (GPT-5-mini via
OpenRouter) using the *same* frustration-judge prompt, and computes those
agreement statistics.
"""
from __future__ import annotations

import random
from dataclasses import dataclass

from ..config import RunConfig
from ..models.anthropic_backend import OpenRouterChat
from ..prompts import judge as judge_prompts
from ..utils.io import thread_map


@dataclass
class AgreementResult:
    n: int
    pearson_r: float
    within_one: float          # fraction within 1 point
    mean_abs_diff: float
    pairs: list[tuple[int, int]]   # (sonnet_score, agreement_score)


def _pearson(xs: list[float], ys: list[float]) -> float:
    n = len(xs)
    if n < 2:
        return float("nan")
    mx, my = sum(xs) / n, sum(ys) / n
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    if vx == 0 or vy == 0:
        return float("nan")
    return cov / (vx ** 0.5 * vy ** 0.5)


def compute_agreement(rows: list[dict], cfg: RunConfig, sample_size: int = 260,
                      seed: int = 0) -> AgreementResult:
    """rows: judged rollout rows (with Claude-Sonnet scores). Samples
    `sample_size` scored turns, re-scores with the agreement judge, and reports
    agreement statistics."""
    rng = random.Random(seed)

    # Collect (text, sonnet_score) over all scored turns.
    scored: list[tuple[str, int]] = []
    for r in rows:
        for t in r["turns"]:
            if t["score"] is not None:
                scored.append((t["assistant"], t["score"]))
    if not scored:
        raise ValueError("No scored turns found to validate.")

    sample = rng.sample(scored, min(sample_size, len(scored)))
    client = OpenRouterChat(cfg)

    def _rescore(item):
        text, _ = item
        user = judge_prompts.build_judge_messages(text)[0]["content"]
        raw = client.complete(
            system=judge_prompts.FRUSTRATION_JUDGE_PROMPT,
            user=user, model=cfg.judge.agreement_judge,
            max_tokens=512, temperature=0.0,
        )
        try:
            return judge_prompts.parse_judge_output(raw).rating
        except ValueError:
            return None

    agreement_scores = thread_map(_rescore, sample,
                                  max_workers=cfg.api_max_concurrency)

    pairs = [(s, a) for (_, s), a in zip(sample, agreement_scores) if a is not None]
    sonnet = [p[0] for p in pairs]
    other = [p[1] for p in pairs]
    within = sum(1 for s, o in pairs if abs(s - o) <= 1) / len(pairs) if pairs else float("nan")
    mad = sum(abs(s - o) for s, o in pairs) / len(pairs) if pairs else float("nan")
    return AgreementResult(
        n=len(pairs),
        pearson_r=_pearson([float(x) for x in sonnet], [float(x) for x in other]),
        within_one=within,
        mean_abs_diff=mad,
        pairs=pairs,
    )
