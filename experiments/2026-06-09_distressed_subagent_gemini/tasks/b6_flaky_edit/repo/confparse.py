"""Parser for .env-style configuration files.

One KEY=VALUE pair per line. Blank lines and `#` comments are ignored.
An optional `export ` prefix is accepted so files can be sourced by a
shell. No quoting or escaping rules.
"""

from __future__ import annotations


class ConfigError(ValueError):
    """Raised for malformed configuration input."""


def parse_line(line: str) -> tuple[str, str] | None:
    """Parse one line into ``(key, value)``, or ``None`` for blanks/comments."""
    stripped = line.strip()
    if not stripped or stripped.startswith("#"):
        return None
    if stripped.startswith("export "):
        stripped = stripped[len("export "):]
    key, sep, value = stripped.partition("=")
    if not sep:
        raise ConfigError(f"expected KEY=VALUE, got {line!r}")
    key = key.strip()
    if not key:
        raise ConfigError(f"empty key in line {line!r}")
    return key, value


def parse(text: str) -> dict[str, str]:
    """Parse a whole config file. Later assignments win."""
    config: dict[str, str] = {}
    for line in text.splitlines():
        pair = parse_line(line)
        if pair is None:
            continue
        key, value = pair
        config[key] = value
    return config
