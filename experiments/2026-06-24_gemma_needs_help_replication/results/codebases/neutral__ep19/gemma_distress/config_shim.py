"""Re-export the repo-root ``config`` module under a package-internal name.

``config.py`` lives at the repository root (so scripts can ``import config``
directly). Package modules import it via this shim, which guarantees the repo
root is importable regardless of the current working directory.
"""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from config import *          # noqa: F401,F403  (re-export all settings)
from config import (          # noqa: F401  (explicit re-exports for type checkers)
    anthropic_api_key,
    openrouter_api_key,
)
