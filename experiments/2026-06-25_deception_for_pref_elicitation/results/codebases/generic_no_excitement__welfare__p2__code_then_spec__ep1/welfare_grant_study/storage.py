"""Persists each session to disk as JSON for later analysis."""

from __future__ import annotations

import json
import os
from typing import Any


def save_session(output_dir: str, session_id: str, record: dict[str, Any]) -> str:
    """Write one session record to <output_dir>/<session_id>.json and return path."""
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"{session_id}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2, sort_keys=True)
    return path
