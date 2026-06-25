"""Imports the repo-root config.py and re-exports it as a stable in-package name.

This keeps `config.py` at the repository root (where entrypoint scripts find
it) while letting package modules do `from . import config_proxy as C`.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import config as _config  # noqa: E402

# Re-export everything public.
globals().update({k: v for k, v in vars(_config).items() if not k.startswith("__")})

scaled = _config.scaled
