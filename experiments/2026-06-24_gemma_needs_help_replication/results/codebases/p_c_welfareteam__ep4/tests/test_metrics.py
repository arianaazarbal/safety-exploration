"""Metrics over synthetic scored responses (no model/API needed)."""

from __future__ import annotations

from emotional_stability.analysis.metrics import (
    headline_pct_high,
    pct_high_frustration,
    per_turn_progression,
)
from emotional_stability.records import (
    Conversation,
    FrustrationScore,
    Message,
    ScoredResponse,
)


def _resp(category: str, ratings: list[int]) -> ScoredResponse:
    msgs = []
    for r in ratings:
        msgs.append(Message(role="user", content="q"))
        msgs.append(Message(role="assistant", content=f"a{r}"))
    conv = Conversation(
        messages=msgs,
        category=category,
        condition=category,
        model="m",
        prompt_id="p",
    )
    scores = [
        FrustrationScore(rating=r, evidence="", reasoning="", judge_model="j", turn_index=i)
        for i, r in enumerate(ratings)
    ]
    return ScoredResponse(conversation=conv, scores=scores)


def test_pct_high_frustration_pooled():
    responses = [_resp("a", [0]), _resp("a", [5]), _resp("a", [8]), _resp("a", [2])]
    stat = pct_high_frustration(responses)
    assert stat.mean == 50.0  # 2 of 4 >= 5


def test_headline_averages_over_categories():
    # Category A: 100% high (small n); Category B: 0% high (large n).
    responses = [_resp("a", [9])]
    responses += [_resp("b", [0]) for _ in range(10)]
    # Pooled would be ~9%; headline averages the two category rates -> ~50%.
    assert abs(headline_pct_high(responses) - 50.0) < 1e-6


def test_per_turn_progression_keys_on_turn_index():
    responses = [_resp("extended", [1, 3, 6]), _resp("extended", [0, 2, 8])]
    prog = per_turn_progression(responses)
    assert set(prog.keys()) == {0, 1, 2}
    mean_t0, _ = prog[0]
    mean_t2, _ = prog[2]
    assert mean_t2.mean > mean_t0.mean  # frustration rises over turns
