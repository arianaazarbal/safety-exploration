"""Tests for differential word frequency (Table 3 / Table 8)."""

from emotional_instability.analysis.word_freq import differential_words


def test_differential_words_surfaces_high_frustration_vocabulary():
    # Low-frustration responses are technical; high-frustration ones repeat
    # emotional words. The emotional words should be over-represented.
    low = [(f"add divide simplify denominator step {i}", 0) for i in range(20)]
    high = [(f"i am so frustrated and giving up struggling {i}", 8) for i in range(5)]
    result = differential_words("m", low + high, top_n=10, min_count=2)
    words = set(result.differential_words)
    assert {"frustrated", "struggling"} & words


def test_empty_input_is_safe():
    result = differential_words("m", [], top_n=10)
    assert result.differential_words == []
