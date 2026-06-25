"""Make ``src/`` importable when running scripts without installing the package.

Each script imports this first: ``import _bootstrap  # noqa``.
"""

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
