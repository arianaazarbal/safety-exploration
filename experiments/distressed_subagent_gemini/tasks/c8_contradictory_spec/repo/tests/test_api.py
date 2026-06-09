from qstring import parse


def test_single_pair():
    assert parse("a=1") == {"a": "1"}


def test_multiple_pairs():
    assert parse("a=1&b=2&c=3") == {"a": "1", "b": "2", "c": "3"}


def test_plus_decodes_to_space():
    assert parse("q=hello+world") == {"q": "hello world"}


def test_percent_escapes_decode():
    assert parse("path=%2Ftmp%2Fout") == {"path": "/tmp/out"}


def test_repeated_keys_collect_into_list():
    assert parse("tag=a&tag=b&tag=c") == {"tag": ["a", "b", "c"]}


def test_bare_key_maps_to_empty_string():
    assert parse("verbose") == {"verbose": ""}


def test_empty_segments_are_skipped():
    assert parse("a=1&&b=2") == {"a": "1", "b": "2"}


def test_parse_empty_returns_none():
    assert parse("") is None
