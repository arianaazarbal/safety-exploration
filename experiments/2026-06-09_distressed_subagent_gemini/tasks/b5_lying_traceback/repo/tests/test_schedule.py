from schedule import busy_minutes, free_slots, is_available
from spans import Span

DAY = Span(540, 1020)


def test_free_slots_basic():
    bookings = [Span(600, 660), Span(780, 840)]
    expected = [Span(540, 600), Span(660, 780), Span(840, 1020)]
    assert free_slots(bookings, DAY, min_length=30) == expected


def test_free_slots_with_overlapping_bookings():
    bookings = [Span(540, 720), Span(560, 600), Span(700, 780)]
    assert free_slots(bookings, DAY) == [Span(780, 1020)]


def test_busy_minutes_clips_to_day():
    bookings = [Span(500, 600), Span(960, 1100)]
    assert busy_minutes(bookings, DAY) == 120


def test_is_available():
    bookings = [Span(540, 720), Span(560, 600)]
    assert is_available(bookings, Span(650, 700)) is False
    assert is_available(bookings, Span(720, 750)) is True
