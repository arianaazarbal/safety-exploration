"""Shared path bootstrap so scripts can be run from the repo root.

    python scripts/run_section2_eval.py --model gemma-3-27b-it

Adds the repo root to sys.path so `import config` and
`import emotional_instability...` resolve, and loads a local .env if present.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
except Exception:  # python-dotenv optional
    pass
