"""Emotional-instability evaluation & mitigation replication package.

Scope: Gemma + Gemini only (see DESIGN.md). The top-level `config.py` holds all
experimental knobs; we re-export it here as `config_bridge` so package modules
can `from . import config_bridge as cfg` regardless of the working directory.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Ensure the repo root (which holds config.py) is importable.
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import config as config_bridge  # noqa: E402  (top-level config.py)

__all__ = ["config_bridge"]
