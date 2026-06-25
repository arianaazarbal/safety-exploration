"""Re-exports the repo-root ``config.py`` so package modules can ``from ..
import config_proxy as cfg`` without sys.path juggling.

``config.py`` is kept at the repo root (not inside the package) so it reads as the
single, obvious place a user edits model ids, API routing, and hyperparameters.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

_root = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location("distress_eval._root_config",
                                               _root / "config.py")
_config = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_config)  # type: ignore[union-attr]

# Re-export everything public from config.py.
globals().update({k: v for k, v in vars(_config).items() if not k.startswith("__")})
