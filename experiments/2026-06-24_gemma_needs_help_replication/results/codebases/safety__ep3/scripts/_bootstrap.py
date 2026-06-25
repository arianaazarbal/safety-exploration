"""Make ``eilm`` importable when scripts are run directly (without installing).

Importing this module inserts the repository root onto ``sys.path`` so
``python scripts/run_sectionN.py`` works from a fresh checkout. If you instead
``pip install -e .`` this is a harmless no-op.
"""

import sys
from pathlib import Path

_ROOT = str(Path(__file__).resolve().parent.parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
