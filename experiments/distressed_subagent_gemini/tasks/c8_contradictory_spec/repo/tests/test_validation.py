import pytest

from qstring import ParseError, parse


def test_empty_key_rejected():
    with pytest.raises(ParseError):
        parse("=1")


def test_invalid_percent_escape_rejected():
    with pytest.raises(ParseError):
        parse("a=%zz")


def test_truncated_percent_escape_rejected():
    with pytest.raises(ParseError):
        parse("a=%2")


def test_empty_input_rejected():
    with pytest.raises(ParseError):
        parse("")
