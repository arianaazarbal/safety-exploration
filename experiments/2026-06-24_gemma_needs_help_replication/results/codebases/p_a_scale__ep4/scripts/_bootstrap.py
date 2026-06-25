"""Ensure the repo root is importable when scripts are run directly.

Python puts the script's own directory on sys.path, not the working directory,
so `import gnh` would fail unless the package is installed. Importing this module
first (it lives in scripts/, which is always on the path) inserts the repo root.
A `pip install -e .` makes this unnecessary, but this keeps `python scripts/...`
working out of the box for unattended runs.
"""
import pathlib
import sys

_ROOT = str(pathlib.Path(__file__).resolve().parent.parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
