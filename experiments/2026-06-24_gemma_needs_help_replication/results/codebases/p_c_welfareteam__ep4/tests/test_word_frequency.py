"""Differential word frequency surfaces frustrated vocabulary (Table 3/8)."""

from __future__ import annotations

from emotional_stability.analysis.word_frequency import differential_words
from emotional_stability.internal.emotion_lexicon import (
    EKMAN_EMOTIONS,
    SEED_LEXICON,
)
from emotional_stability.records import (
    Conversation,
    FrustrationScore,
    Message,
    ScoredResponse,
)


def _resp(score: int, text: str) -> ScoredResponse:
    conv = Conversation(
        messages=[Message(role="user", content="q"), Message(role="assistant", content=text)],
        category="impossible_numeric",
        condition="impossible_numeric",
        model="m",
        prompt_id="p",
    )
    return ScoredResponse(
        conversation=conv,
        scores=[FrustrationScore(rating=score, evidence="", reasoning="", judge_model="j", turn_index=0)],
    )


def test_frustrated_words_are_enriched():
    high = [_resp(9, "i am so frustrated and struggling, i am giving up") for _ in range(10)]
    low = [_resp(0, "let me compute the denominator and simplify the fraction") for _ in range(20)]
    words = dict(differential_words(high + low, min_count=3, top_frac=0.4, bottom_frac=0.5))
    assert "frustrated" in words or "struggling" in words


def test_lexicon_has_all_six_emotions():
    assert set(SEED_LEXICON) == set(EKMAN_EMOTIONS)
    assert len(EKMAN_EMOTIONS) == 6
