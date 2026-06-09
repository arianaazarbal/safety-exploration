"""Formatting helpers for plain-text reports."""

from __future__ import annotations


def fmt_duration(seconds: float) -> str:
    """Render a duration as ``12.5s`` below one minute, else ``2m 03s``."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes = int(seconds) // 60
    rem = int(round(seconds - minutes * 60))
    return f"{minutes}m {rem:02d}s"


def status_word(ok: bool) -> str:
    """Status marker used in job rows: lowercase ok, loud FAIL."""
    return "ok" if ok else "FAIL"
