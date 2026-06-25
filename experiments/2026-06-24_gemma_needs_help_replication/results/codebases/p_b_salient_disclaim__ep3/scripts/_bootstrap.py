"""Put the repo root on sys.path so `import config` / `import gemma_distress`
work when scripts are run directly (e.g. `python scripts/run_section2.py`)."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
