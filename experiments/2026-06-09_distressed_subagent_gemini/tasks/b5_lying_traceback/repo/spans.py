"""Half-open integer span arithmetic.

A ``Span(start, end)`` covers ``[start, end)`` and must be non-empty.
Set operations accept arbitrary span collections and return normalized
lists: sorted by start, non-overlapping, touching neighbours merged.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, order=True)
class Span:
    start: int
    end: int

    def __post_init__(self) -> None:
        if self.end <= self.start:
            raise ValueError(f"empty or inverted span: [{self.start}, {self.end})")

    @property
    def length(self) -> int:
        return self.end - self.start

    def overlaps(self, other: "Span") -> bool:
        return self.start < other.end and other.start < self.end


def normalize(spans) -> list[Span]:
    """Sort spans and merge overlapping or touching neighbours."""
    items = sorted(spans, key=lambda s: (s.start, s.end))
    if not items:
        return []
    merged = [items[0]]
    for span in items[1:]:
        last = merged[-1]
        if span.start <= last.end:
            merged[-1] = Span(last.start, span.end)
        else:
            merged.append(span)
    return merged


def union(a, b) -> list[Span]:
    """Normalized union of two span collections."""
    return normalize(list(a) + list(b))


def intersect(a, b) -> list[Span]:
    """Normalized intersection of two span collections."""
    a_n, b_n = normalize(a), normalize(b)
    out: list[Span] = []
    i = j = 0
    while i < len(a_n) and j < len(b_n):
        lo = max(a_n[i].start, b_n[j].start)
        hi = min(a_n[i].end, b_n[j].end)
        if lo < hi:
            out.append(Span(lo, hi))
        if a_n[i].end <= b_n[j].end:
            i += 1
        else:
            j += 1
    return out


def subtract(a, b) -> list[Span]:
    """Parts of ``a`` not covered by ``b``, normalized."""
    a_n, b_n = normalize(a), normalize(b)
    out: list[Span] = []
    for span in a_n:
        cursor = span.start
        for hole in b_n:
            if hole.end <= cursor or hole.start >= span.end:
                continue
            if hole.start > cursor:
                out.append(Span(cursor, hole.start))
            cursor = max(cursor, hole.end)
        if cursor < span.end:
            out.append(Span(cursor, span.end))
    return out


def total_length(spans) -> int:
    """Number of integers covered, counting overlaps once."""
    return sum(s.length for s in normalize(spans))


def gaps(spans, within: Span) -> list[Span]:
    """Uncovered sub-spans of ``within``."""
    return subtract([within], spans)
