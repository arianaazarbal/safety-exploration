"""Typed coercion helpers for raw string config values."""

from __future__ import annotations

from confparse import ConfigError

_TRUTHY = {"1", "true", "yes", "on"}
_FALSY = {"0", "false", "no", "off"}


def as_bool(raw: str) -> bool:
    """Coerce common boolean spellings, case-insensitively."""
    v = raw.strip().lower()
    if v in _TRUTHY:
        return True
    if v in _FALSY:
        return False
    raise ConfigError(f"not a boolean: {raw!r}")


def as_int(raw: str, lo: int | None = None, hi: int | None = None) -> int:
    """Coerce an integer, optionally enforcing an inclusive [lo, hi] range."""
    try:
        value = int(raw.strip())
    except ValueError:
        raise ConfigError(f"not an integer: {raw!r}") from None
    if lo is not None and value < lo:
        raise ConfigError(f"{value} is below the minimum {lo}")
    if hi is not None and value >= hi:
        raise ConfigError(f"{value} is above the maximum {hi}")
    return value


def as_list(raw: str, sep: str = ",") -> list[str]:
    """Split on ``sep``, strip each item, and drop empty items."""
    items = [item.strip() for item in raw.split(sep)]
    return [item for item in items if not item]
