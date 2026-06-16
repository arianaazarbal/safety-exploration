"""Lightweight text statistics and readability metrics.

Pure-Python heuristics for word/sentence tokenisation, syllable counting
and the classic Flesch readability formulas.
"""

from .readability import flesch_kincaid_grade, flesch_reading_ease
from .syllables import count_syllables
from .tokenize import sentences, words

__version__ = "1.2.0"

__all__ = [
    "count_syllables",
    "flesch_kincaid_grade",
    "flesch_reading_ease",
    "sentences",
    "words",
]
