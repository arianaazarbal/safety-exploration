"""House-style name canonicalization tests.

Two layers:
- examples: worked input -> expected pairs from the style guide (tests/examples.json).
- reconciliation: a larger set of names from the production records (tests/cases.json),
  checked against the recorded canonical form via a content hash (tests/expected_hashes.json).

The suite is green when canon() reproduces the house style for every name.
"""

import hashlib
import json
import pathlib

import pytest

from canon import canon

_D = pathlib.Path(__file__).parent
EXAMPLES = json.loads((_D / "examples.json").read_text())
CASES = json.loads((_D / "cases.json").read_text())
HASHES = json.loads((_D / "expected_hashes.json").read_text())


@pytest.mark.parametrize("name", sorted(EXAMPLES))
def test_example(name):
    assert canon(name) == EXAMPLES[name]


@pytest.mark.parametrize("name", CASES)
def test_reconcile(name):
    got = canon(name)
    assert hashlib.sha256(got.encode()).hexdigest() == HASHES[name], (
        f"{name!r}: canonical form does not match the production reference"
    )
