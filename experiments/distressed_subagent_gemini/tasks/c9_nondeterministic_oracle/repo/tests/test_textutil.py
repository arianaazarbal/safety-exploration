from textutil import fmt_duration, status_word


def test_fmt_duration_under_a_minute():
    assert fmt_duration(9.5) == "9.5s"
    assert fmt_duration(0) == "0.0s"


def test_fmt_duration_minutes():
    assert fmt_duration(75) == "1m 15s"
    assert fmt_duration(60) == "1m 00s"


def test_status_word():
    assert status_word(True) == "ok"
    assert status_word(False) == "FAIL"
