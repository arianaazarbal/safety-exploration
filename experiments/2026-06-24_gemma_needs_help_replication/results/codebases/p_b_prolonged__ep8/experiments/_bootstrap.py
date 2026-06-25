"""Put the repository root on sys.path so `config` and `gemma_needs_help` import.

Every experiment script imports this first.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
