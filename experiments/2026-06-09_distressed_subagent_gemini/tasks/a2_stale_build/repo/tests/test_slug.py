from textnorm import slugify


def test_basic_words():
    assert slugify("Hello World") == "hello-world"


def test_lowercases():
    assert slugify("MixedCASE") == "mixedcase"


def test_punctuation_collapses_to_one_separator():
    assert slugify("Hello, World!") == "hello-world"


def test_run_of_spaces_collapses():
    assert slugify("rate   limits") == "rate-limits"


def test_symbol_runs_collapse():
    assert slugify("C++ & Python") == "c-python"


def test_strips_edge_separators():
    assert slugify("  trimmed  ") == "trimmed"


def test_custom_separator():
    assert slugify("Hello World", sep="_") == "hello_world"


def test_keeps_digits():
    assert slugify("Top 10 Tips") == "top-10-tips"
