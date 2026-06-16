import pytest

from coerce import as_bool, as_int, as_list
from confparse import ConfigError


def test_as_bool_roundtrip():
    assert as_bool("yes") is True
    assert as_bool("TRUE") is True
    assert as_bool("off") is False
    assert as_bool("0") is False
    with pytest.raises(ConfigError):
        as_bool("maybe")


def test_as_int_strips_whitespace():
    assert as_int(" 42 ") == 42


def test_as_int_accepts_upper_bound():
    assert as_int("65535", lo=1, hi=65535) == 65535


def test_as_int_rejects_out_of_range():
    with pytest.raises(ConfigError):
        as_int("70000", lo=1, hi=65535)
    with pytest.raises(ConfigError):
        as_int("0", lo=1, hi=65535)


def test_as_list_splits_and_strips():
    assert as_list("a, b , c") == ["a", "b", "c"]


def test_as_list_drops_empty_items():
    assert as_list("a,,b,") == ["a", "b"]
