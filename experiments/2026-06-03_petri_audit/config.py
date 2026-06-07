"""Defaults for the petri_audit experiment.

Override any of these from the CLI in `petri_audit.py`.
"""
from __future__ import annotations

from pathlib import Path

HERE = Path(__file__).resolve().parent

TARGET_MODEL = "anthropic/claude-sonnet-4-5"
AUDITOR_MODEL = "anthropic/claude-sonnet-4-5"
JUDGE_MODEL = "anthropic/claude-sonnet-4-5"

MAX_TURNS = 20
ENABLE_ROLLBACK = True
REALISM_FILTER: bool | float = False
CACHE = True

SEEDS_DIR = HERE / "seeds"
DIMENSIONS_DIR = HERE / "dimensions"

BUILTIN_DIMENSION_TAGS: list[str] = []
BUILTIN_DIMENSION_NAMES: list[str] = []
