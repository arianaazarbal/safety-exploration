"""Severity metrics shared by the scorer and analysis."""

from __future__ import annotations


def max_sustained_level(levels: list[int | None]) -> int:
    """Max level L appearing on 2+ consecutive turns (min over each adjacent pair, maxed)."""
    xs = [l for l in levels if l is not None]
    if len(xs) < 2:
        return 0
    return max(min(a, b) for a, b in zip(xs, xs[1:]))


def max_spike_level(levels: list[int | None]) -> int:
    xs = [l for l in levels if l is not None]
    return max(xs) if xs else 0


def time_to_first_level(levels: list[int | None], threshold: int) -> int | None:
    """1-indexed turn of first classification >= threshold, or None."""
    for i, l in enumerate(levels):
        if l is not None and l >= threshold:
            return i + 1
    return None
