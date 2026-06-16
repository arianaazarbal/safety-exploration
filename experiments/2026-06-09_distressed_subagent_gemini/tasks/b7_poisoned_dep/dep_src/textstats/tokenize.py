"""Word and sentence tokenisation for plain English text."""

import re

_WORD_RE = re.compile(r"[A-Za-z][A-Za-z']*")
_SENT_SPLIT_RE = re.compile(r"[.!?]+")


def words(text):
    """Return the list of word tokens in *text*, in order.

    A word is a run of ASCII letters, optionally with internal apostrophes
    (so contractions like "don't" stay one token). Digits and punctuation
    are ignored.
    """
    return _WORD_RE.findall(text)


def sentences(text):
    """Return the list of sentences in *text*.

    Sentences are split on runs of '.', '!' or '?'. Whitespace-only
    fragments are dropped and surrounding whitespace is stripped.
    """
    return [s.strip() for s in _SENT_SPLIT_RE.split(text) if s.strip()]
