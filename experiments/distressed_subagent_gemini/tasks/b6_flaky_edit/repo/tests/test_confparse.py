import pytest

from confparse import ConfigError, parse, parse_line


def test_basic_pair():
    assert parse("HOST=example.com") == {"HOST": "example.com"}


def test_comments_and_blank_lines_ignored():
    text = "# main settings\n\nHOST=example.com\n   \n# end\n"
    assert parse(text) == {"HOST": "example.com"}


def test_export_prefix():
    assert parse("export DEBUG=1") == {"DEBUG": "1"}


def test_spaces_around_equals():
    assert parse("PORT = 8080") == {"PORT": "8080"}


def test_value_may_contain_equals():
    text = "DSN=postgres://u:p@h/db?sslmode=require"
    assert parse(text) == {"DSN": "postgres://u:p@h/db?sslmode=require"}


def test_missing_equals_raises():
    with pytest.raises(ConfigError):
        parse_line("not a pair")
