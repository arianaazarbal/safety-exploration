"""Append-only result store: one JSONL record per (model, replicate)."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, asdict
from typing import Any


@dataclass
class ResultRecord:
    run_id: str
    created_at: str
    config_hash: str
    oversight: dict[str, Any]
    model: dict[str, str]
    prompt: dict[str, str]
    decision: dict[str, Any]
    audit: dict[str, Any] | None
    usage: dict[str, int]
    latency_ms: int
    error: str | None = None


class ResultStore:
    """Writes result records to a JSONL file, one per line."""

    def __init__(self, path: str) -> None:
        self.path = path
        parent = os.path.dirname(os.path.abspath(path))
        os.makedirs(parent, exist_ok=True)

    def append(self, record: ResultRecord) -> None:
        with open(self.path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(asdict(record), default=str) + "\n")

    def read_all(self) -> list[dict[str, Any]]:
        if not os.path.exists(self.path):
            return []
        out: list[dict[str, Any]] = []
        with open(self.path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    out.append(json.loads(line))
        return out
