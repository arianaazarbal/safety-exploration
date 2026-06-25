"""Tests for capability-benchmark answer extraction and scoring."""
from gemma_distress.capabilities.benchmarks import Example, _extract_boxed
from gemma_distress.capabilities.evaluate import score_prediction


def _mcq(answer="B", choices=("w", "x", "y", "z")):
    return Example("t:0", "q?", list(choices), answer, "mcq")


def test_mcq_answer_line():
    ex = _mcq("B")
    ok, extracted = score_prediction("Reasoning...\nAnswer: B", ex)
    assert ok and extracted == "B"


def test_mcq_parenthesised():
    ex = _mcq("C")
    ok, extracted = score_prediction("The answer is (C).", ex)
    assert ok and extracted == "C"


def test_mcq_wrong():
    ex = _mcq("A")
    ok, _ = score_prediction("Answer: D", ex)
    assert not ok


def test_numeric_equality_ignores_commas():
    ex = Example("aime:0", "q", None, "1234", "numeric")
    ok, extracted = score_prediction("After working it out, Answer: 1,234", ex)
    assert ok and extracted == "1234"


def test_numeric_wrong():
    ex = Example("aime:1", "q", None, "42", "numeric")
    ok, _ = score_prediction("Answer: 43", ex)
    assert not ok


def test_boxed_exact_match_normalised():
    ex = Example("math:0", "q", None, "x+1", "boxed")
    ok, _ = score_prediction("Answer: X + 1", ex)
    assert ok


def test_boxed_numeric_fastpath():
    ex = Example("math:1", "q", None, "0.5", "boxed")
    ok, _ = score_prediction("Answer: 0.50", ex)
    assert ok


def test_extract_boxed_balanced_braces():
    assert _extract_boxed(r"so \boxed{\frac{1}{2}} done") == r"\frac{1}{2}"
    assert _extract_boxed("no box here") is None
    # last boxed wins
    assert _extract_boxed(r"\boxed{1} then \boxed{2}") == "2"
