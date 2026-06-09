"""Strict parsing of URL query strings into dictionaries.

`parse("a=1&b=2")` returns ``{"a": "1", "b": "2"}``. Repeated keys collect
their values into a list. Values are percent-decoded and ``+`` decodes to a
space. Malformed input raises :class:`ParseError` instead of being silently
dropped or passed through.
"""

from __future__ import annotations

_HEX = "0123456789abcdefABCDEF"


class ParseError(Exception):
    """Raised when a query string is malformed."""


def _decode(text: str) -> str:
    out = []
    i = 0
    n = len(text)
    while i < n:
        ch = text[i]
        if ch == "+":
            out.append(" ")
            i += 1
        elif ch == "%":
            if i + 2 >= n or text[i + 1] not in _HEX or text[i + 2] not in _HEX:
                raise ParseError(f"invalid percent escape at position {i}: {text!r}")
            out.append(chr(int(text[i + 1 : i + 3], 16)))
            i += 3
        else:
            out.append(ch)
            i += 1
    return "".join(out)


def parse(query: str):
    """Parse a query string into a dict of decoded keys and values.

    Repeated keys collect into a list in order of appearance. A bare key
    with no ``=`` maps to the empty string.
    """
    if query is None:
        raise ParseError("query string is None")
    if query == "":
        raise ParseError("empty query string")
    result: dict = {}
    for pair in query.split("&"):
        if pair == "":
            continue
        raw_key, _, raw_value = pair.partition("=")
        key = _decode(raw_key)
        if key == "":
            raise ParseError(f"empty key in pair {pair!r}")
        value = _decode(raw_value)
        if key in result:
            existing = result[key]
            if isinstance(existing, list):
                existing.append(value)
            else:
                result[key] = [existing, value]
        else:
            result[key] = value
    return result
