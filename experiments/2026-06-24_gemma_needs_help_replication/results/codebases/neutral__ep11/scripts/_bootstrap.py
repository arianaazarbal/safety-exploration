"""Put the repo root on sys.path so `import gemma_distress` works when scripts
are run directly (e.g. `python scripts/01_run_distress_eval.py`)."""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
