import pytest

from spans import Span, gaps, normalize, subtract, total_length


def test_span_rejects_empty():
    with pytest.raises(ValueError):
        Span(5, 5)


def test_normalize_sorts_disjoint():
    assert normalize([Span(10, 12), Span(0, 3)]) == [Span(0, 3), Span(10, 12)]


def test_normalize_merges_overlap():
    assert normalize([Span(0, 5), Span(3, 8)]) == [Span(0, 8)]


def test_normalize_merges_touching():
    assert normalize([Span(0, 5), Span(5, 9)]) == [Span(0, 9)]


def test_normalize_handles_contained_span():
    spans = [Span(0, 10), Span(2, 4), Span(6, 12)]
    assert normalize(spans) == [Span(0, 12)]


def test_subtract_punches_hole():
    assert subtract([Span(0, 10)], [Span(3, 5)]) == [Span(0, 3), Span(5, 10)]


def test_total_length_counts_overlap_once():
    assert total_length([Span(1, 9), Span(2, 3), Span(4, 5)]) == 8


def test_gaps_empty_when_fully_covered():
    covering = [Span(0, 9), Span(2, 4), Span(6, 9)]
    assert gaps(covering, Span(0, 9)) == []
