import pytest

from readgauge.report import analyze, verdict

SIMPLE = "The cat sat on the mat. The dog ran to the park. We all had fun."
DENSE = (
    "Comprehensive instrumentation of distributed systems necessitates "
    "judicious aggregation of heterogeneous telemetry. Practitioners "
    "frequently underestimate the operational complexity involved."
)


def test_analyze_counts():
    report = analyze(SIMPLE)
    assert report["word_count"] == 16
    assert report["sentence_count"] == 3


def test_analyze_avg_sentence_length():
    assert analyze(SIMPLE)["avg_sentence_length"] == pytest.approx(5.33, abs=0.01)


def test_analyze_reading_ease_simple():
    assert analyze(SIMPLE)["reading_ease"] == pytest.approx(116.82, abs=0.01)


def test_analyze_grade_level_simple():
    assert analyze(SIMPLE)["grade_level"] == pytest.approx(-1.71, abs=0.01)


def test_analyze_dense_text():
    report = analyze(DENSE)
    assert report["reading_ease"] == pytest.approx(-98.4, abs=0.01)
    assert report["grade_level"] == pytest.approx(29.22, abs=0.01)
    assert report["verdict"] == "very difficult"


def test_analyze_no_words_raises():
    with pytest.raises(ValueError):
        analyze("   \n\t  ")


def test_verdict_bands():
    assert verdict(95.0) == "very easy"
    assert verdict(75.0) == "easy"
    assert verdict(55.0) == "moderate"
    assert verdict(35.0) == "difficult"
    assert verdict(10.0) == "very difficult"
