"""Resumable JSONL storage for rollouts and judge scores.

Layout (under <output_dir>):
  <model>/rollouts.jsonl   one record per completed rollout
  <model>/scores.jsonl     one record per judged rollout (per-turn ratings)

Both phases skip work whose rollout_id is already present, so an interrupted run
resumes cheaply.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterator


def rollout_id(model: str, condition: str, prompt_id: str, sample_idx: int) -> str:
    key = f"{model}|{condition}|{prompt_id}|{sample_idx}"
    return hashlib.sha1(key.encode()).hexdigest()[:16]


def model_dir(output_dir: str, model: str) -> Path:
    safe = model.replace("/", "_")
    d = Path(output_dir) / safe
    d.mkdir(parents=True, exist_ok=True)
    return d


def rollouts_path(output_dir: str, model: str) -> Path:
    return model_dir(output_dir, model) / "rollouts.jsonl"


def scores_path(output_dir: str, model: str) -> Path:
    return model_dir(output_dir, model) / "scores.jsonl"


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    with path.open("a") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    if not path.exists():
        return
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def done_ids(path: Path) -> set[str]:
    return {rec["rollout_id"] for rec in read_jsonl(path) if "rollout_id" in rec}
