from textnorm import collapse_whitespace, truncate


def test_collapse_whitespace_runs():
    assert collapse_whitespace("a \t b\n\nc") == "a b c"


def test_collapse_whitespace_trims():
    assert collapse_whitespace("  padded  ") == "padded"


def test_truncate_short_text_unchanged():
    assert truncate("hello", 10) == "hello"


def test_truncate_cuts_at_word_boundary():
    assert truncate("the quick brown fox", 15) == "the quick..."


def test_truncate_custom_ellipsis():
    assert truncate("abcdefghij", 8, ellipsis="~") == "abcdefg~"
