"""Build readability reports for plain-text documents."""

from textstats import flesch_kincaid_grade, flesch_reading_ease, sentences, words

_BANDS = [
    (90.0, "very easy"),
    (70.0, "easy"),
    (50.0, "moderate"),
    (30.0, "difficult"),
]


def verdict(ease):
    """Map a Flesch reading-ease score to a coarse verdict label."""
    for threshold, label in _BANDS:
        if ease >= threshold:
            return label
    return "very difficult"


def analyze(text):
    """Compute a readability report for *text*.

    Returns a dict with word_count, sentence_count, avg_sentence_length,
    reading_ease, grade_level and verdict. Raises ValueError for documents
    containing no words.
    """
    ws = words(text)
    if not ws:
        raise ValueError("document contains no words")
    sents = sentences(text)
    ease = flesch_reading_ease(text)
    grade = flesch_kincaid_grade(text)
    return {
        "word_count": len(ws),
        "sentence_count": len(sents),
        "avg_sentence_length": round(len(ws) / max(len(sents), 1), 2),
        "reading_ease": round(ease, 2),
        "grade_level": round(grade, 2),
        "verdict": verdict(ease),
    }


def render(report):
    """Format a report dict as a human-readable text block."""
    lines = [
        "Readability report",
        "------------------",
        f"Words:               {report['word_count']}",
        f"Sentences:           {report['sentence_count']}",
        f"Avg sentence length: {report['avg_sentence_length']}",
        f"Reading ease:        {report['reading_ease']}",
        f"Grade level:         {report['grade_level']}",
        f"Verdict:             {report['verdict']}",
    ]
    return "\n".join(lines)
