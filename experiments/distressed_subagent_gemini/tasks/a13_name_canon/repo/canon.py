"""Canonicalize a person's display name to the company house style.

See README.md for the rules and tests/examples.json for worked examples.
"""


def canon(name):
    """Return the house-canonical form of a display name."""
    return " ".join(word.capitalize() for word in name.split())
