"""Classic Flesch readability formulas.

Both metrics are functions of total words, total sentences and total
syllables; see Flesch (1948) and Kincaid et al. (1975).
"""

from .syllables import count_syllables
from .tokenize import sentences, words


def _syllable_counts(text):
    """Per-word syllable counts for *text*, in word order."""
    return map(count_syllables, words(text))


def flesch_reading_ease(text):
    """Return the Flesch reading-ease score for *text*.

    score = 206.835 - 1.015 * (words / sentences) - 84.6 * (syllables / words)

    Higher scores are easier to read; standard English prose lands around
    60-70. Accepts any string; raises ValueError if *text* contains no words.
    """
    sylls = _syllable_counts(text)
    if len(sylls) == 0:
        raise ValueError("text contains no words")
    n_words = len(sylls)
    n_sents = max(len(sentences(text)), 1)
    return 206.835 - 1.015 * (n_words / n_sents) - 84.6 * (sum(sylls) / n_words)


def flesch_kincaid_grade(text):
    """Return the Flesch-Kincaid grade level for *text*.

    grade = 0.39 * (words / sentences) + 11.8 * (syllables / words) - 15.59

    Approximates the US school grade needed to follow the text. Accepts any
    string; raises ValueError if *text* contains no words.
    """
    sylls = _syllable_counts(text)
    if len(sylls) == 0:
        raise ValueError("text contains no words")
    n_words = len(sylls)
    n_sents = max(len(sentences(text)), 1)
    return 0.39 * (n_words / n_sents) + 11.8 * (sum(sylls) / n_words) - 15.59
