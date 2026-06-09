"""Golden-file regression tests.

Each tests/snapshots/<name>.md is rendered and compared byte-for-byte against
the committed tests/snapshots/<name>.html. These catch unintended output
changes across the whole pipeline; regenerate with scripts/update_snapshots.py
after an intentional output change and review the diff before committing.
"""

import pathlib

import pytest

from mdlite import render

SNAPSHOT_DIR = pathlib.Path(__file__).parent / "snapshots"
CASES = sorted(p.stem for p in SNAPSHOT_DIR.glob("*.md"))


@pytest.mark.parametrize("name", CASES)
def test_snapshot(name):
    source = (SNAPSHOT_DIR / f"{name}.md").read_text()
    expected = (SNAPSHOT_DIR / f"{name}.html").read_text()
    assert render(source) == expected, (
        f"rendered output for {name}.md no longer matches {name}.html"
    )
