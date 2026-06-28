"""Append-only JSONL logging of trial records."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any


def append_result(path: str, record: dict[str, Any]) -> None:
    """Append one trial record to a JSONL file, stamping it with a UTC timestamp."""
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    stamped = {"recorded_at": datetime.now(timezone.utc).isoformat(), **record}
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(stamped, default=str, ensure_ascii=False) + "\n")
