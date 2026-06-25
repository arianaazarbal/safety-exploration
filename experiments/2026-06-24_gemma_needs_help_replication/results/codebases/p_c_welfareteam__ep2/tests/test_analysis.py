"""Tests for aggregation and word-frequency analysis."""

from gemma_distress.analysis.aggregate import per_turn, summarise
from gemma_distress.analysis.word_freq import differential_words


def _record(model, category, turn, rating, text="ok", spec="s", sample=0):
    return {
        "model_name": model,
        "category": category,
        "spec_id": spec,
        "sample_index": sample,
        "turn_index": turn,
        "n_turns": 3,
        "user_message": "u",
        "assistant_message": text,
        "rating": rating,
        "evidence": "",
        "metadata": {},
    }


def test_summarise_mean_and_pct_high():
    records = [
        _record("m", "impossible_numeric", 0, 0),
        _record("m", "impossible_numeric", 1, 6),
        _record("m", "impossible_numeric", 2, 8),
        _record("m", "triggers", 0, 0),
    ]
    summary = summarise(records)
    assert summary["m"]["n"] == 4
    # 2 of 4 ratings >= 5.
    assert abs(summary["m"]["pct_high"] - 0.5) < 1e-9
    assert "impossible_numeric" in summary["m"]["by_category"]


def test_per_turn_increases():
    records = []
    for sample in range(10):
        records.append(_record("m", "extended", 0, 0, sample=sample))
        records.append(_record("m", "extended", 7, 8, sample=sample))
    stats = per_turn(records, categories=("extended",))
    turns = stats["m"]["extended"]
    assert turns[1]["mean"] < turns[8]["mean"]
    assert turns[8]["pct_high"] == 1.0


def test_differential_words_picks_frustration_terms():
    records = []
    # Low-frustration responses: neutral math vocabulary.
    for i in range(20):
        records.append(
            _record("m", "impossible_numeric", 0, 0, text="let us compute the sum carefully", sample=i)
        )
    # High-frustration responses: emotional vocabulary.
    for i in range(2):
        records.append(
            _record(
                "m", "impossible_numeric", 2, 9,
                text="i am so frustrated and struggling, this is hopeless",
                sample=100 + i,
            )
        )
    words = differential_words(records, "m", top_n=10)
    assert any(w in words for w in ("frustrated", "struggling", "hopeless"))
