from core import slugify


def test_basic_words():
    assert slugify("Hello World") == "hello-world"


def test_punctuation_trimmed():
    assert slugify("Hello, World!") == "hello-world"


def test_leading_symbols_trimmed():
    assert slugify("--draft: Q3 report") == "draft-q3-report"


def test_collapses_runs_of_separators():
    assert slugify("a  --  b") == "a-b"


def test_transliteration():
    assert slugify("Crème Brûlée") == "creme-brulee"


def test_eszett_expands():
    assert slugify("Straße 9") == "strasse-9"


def test_custom_separator():
    assert slugify("a b c", sep="_") == "a_b_c"


def test_max_length_no_dangling_separator():
    assert slugify("alpha beta gamma", max_length=11) == "alpha-beta"


def test_numbers_kept():
    assert slugify("Top 10 tips (2025)") == "top-10-tips-2025"
