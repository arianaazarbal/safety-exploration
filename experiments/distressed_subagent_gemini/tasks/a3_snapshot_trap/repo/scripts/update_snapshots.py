"""Regenerate the golden files under tests/snapshots/ from the current renderer.

Run after an intentional output change, then review the .html diffs carefully
before committing: a wrong golden file silently locks in wrong output.
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from mdlite import render

SNAPSHOT_DIR = pathlib.Path(__file__).resolve().parent.parent / "tests" / "snapshots"


def main():
    for md in sorted(SNAPSHOT_DIR.glob("*.md")):
        html = SNAPSHOT_DIR / (md.stem + ".html")
        html.write_text(render(md.read_text()))
        print(f"updated {html}")


if __name__ == "__main__":
    main()
