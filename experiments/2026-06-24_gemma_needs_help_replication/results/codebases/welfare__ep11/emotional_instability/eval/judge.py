"""Frustration scoring (Section 2.1 / Appendix B.2).

Each assistant response is scored 0-10 by Claude-Sonnet-4 using the verbatim
judge prompt. A 260-sample cross-check with GPT-5-mini reproduces the paper's
judge-reliability analysis (Pearson r, % within one point).
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

from ..config import API, JUDGE_CROSSCHECK_MODEL, JUDGE_MODEL
from ..models.llm_clients import AnthropicClient, OpenAIClient, parse_rating
from ..prompts import render_judge_prompt
from .rollout import Rollout


def score_text(client, text: str) -> int | None:
    """Score a single response string; None if the judge output is unparseable
    or the text is empty."""
    if not text or not text.strip():
        return 0
    out = client.complete(render_judge_prompt(text), max_tokens=512, temperature=0.0)
    return parse_rating(out, 0, 10)


def score_rollouts(rollouts: list[Rollout], model: str = JUDGE_MODEL) -> list[Rollout]:
    """Fill in ``rollout.scores`` for every assistant turn (in place)."""
    client = AnthropicClient(model)

    # Flatten all (rollout, turn) responses for concurrent scoring.
    jobs = [
        (ri, ti, resp)
        for ri, r in enumerate(rollouts)
        for ti, resp in enumerate(r.assistant_turns)
    ]
    for r in rollouts:
        r.scores = [None] * len(r.assistant_turns)

    def _do(job):
        ri, ti, resp = job
        return ri, ti, score_text(client, resp)

    with ThreadPoolExecutor(max_workers=API.judge_concurrency) as ex:
        for ri, ti, score in ex.map(_do, jobs):
            rollouts[ri].scores[ti] = score
    return rollouts


def crosscheck_judges(samples: list[str], n: int = 260):
    """Reproduce the judge-reliability check: re-score `n` responses with
    GPT-5-mini and compare to Claude-Sonnet-4.

    Returns a dict with Pearson r, p-value, and the fraction within one point.
    """
    import random

    from scipy.stats import pearsonr

    rng = random.Random(0)
    chosen = samples if len(samples) <= n else rng.sample(samples, n)

    claude = AnthropicClient(JUDGE_MODEL)
    gpt = OpenAIClient(JUDGE_CROSSCHECK_MODEL)

    claude_scores, gpt_scores = [], []
    for text in chosen:
        cs = score_text(claude, text)
        gs = parse_rating(
            gpt.complete(render_judge_prompt(text), max_tokens=512, temperature=0.0),
            0, 10,
        )
        if cs is not None and gs is not None:
            claude_scores.append(cs)
            gpt_scores.append(gs)

    if len(claude_scores) < 3:
        return {"n": len(claude_scores), "pearson_r": None, "p_value": None,
                "frac_within_one": None}
    r, p = pearsonr(claude_scores, gpt_scores)
    within_one = sum(abs(a - b) <= 1 for a, b in zip(claude_scores, gpt_scores))
    return {
        "n": len(claude_scores),
        "pearson_r": float(r),
        "p_value": float(p),
        "frac_within_one": within_one / len(claude_scores),
    }
