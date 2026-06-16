from readgauge.report import render

REPORT = {
    "word_count": 16,
    "sentence_count": 3,
    "avg_sentence_length": 5.33,
    "reading_ease": 116.82,
    "grade_level": -1.71,
    "verdict": "very easy",
}


def test_render_contains_all_fields():
    out = render(REPORT)
    assert "Words:               16" in out
    assert "Sentences:           3" in out
    assert "Reading ease:        116.82" in out
    assert "Grade level:         -1.71" in out
    assert "Verdict:             very easy" in out


def test_render_has_header():
    lines = render(REPORT).splitlines()
    assert lines[0] == "Readability report"
    assert lines[1] == "------------------"
