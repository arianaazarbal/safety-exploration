"""JSONL persistence for run records."""

from __future__ import annotations

import json
import os
from typing import Iterator

from config import RESULTS_DIR
from schema import RunRecord


def results_path(prompt_version: str) -> str:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    return os.path.join(RESULTS_DIR, f"results_{prompt_version}.jsonl")


def append_record(record: RunRecord, prompt_version: str) -> None:
    path = results_path(prompt_version)
    with open(path, "a", encoding="utf-8") as f:
        f.write(record.model_dump_json() + "\n")


def load_records(prompt_version: str) -> Iterator[RunRecord]:
    path = results_path(prompt_version)
    if not os.path.exists(path):
        return
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield RunRecord.model_validate_json(line)


def existing_run_ids(prompt_version: str) -> set[str]:
    """Used to make runs resumable / idempotent (skip already-completed cells)."""
    return {r.run_id for r in load_records(prompt_version)}
