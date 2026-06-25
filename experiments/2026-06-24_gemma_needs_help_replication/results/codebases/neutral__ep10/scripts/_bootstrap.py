"""Importing this module puts the repository root on sys.path so the
`emotional_instability` package is importable when a script is run directly
(e.g. `python scripts/run_elicitation.py`) without `pip install -e .`."""

import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
