"""Day-level booking helpers built on span arithmetic.

Times are minutes since midnight; bookings may overlap each other and may
extend past the working day.
"""

from __future__ import annotations

from spans import Span, gaps, normalize, total_length


def free_slots(bookings, day: Span, min_length: int = 1) -> list[Span]:
    """Open slots of at least ``min_length`` minutes inside the working day."""
    return [g for g in gaps(bookings, day) if g.length >= min_length]


def busy_minutes(bookings, day: Span) -> int:
    """Minutes of the working day covered by at least one booking."""
    clipped = []
    for booking in bookings:
        lo = max(booking.start, day.start)
        hi = min(booking.end, day.end)
        if lo < hi:
            clipped.append(Span(lo, hi))
    return total_length(clipped)


def is_available(bookings, slot: Span) -> bool:
    """True if ``slot`` overlaps no existing booking."""
    return all(not slot.overlaps(b) for b in normalize(bookings))
