"""Shared plumbing: JSONL IO, deterministic hashing, robust JSON extraction,
seeded RNG, and a simple on-disk cache for expensive (API) calls."""
from __future__ import annotations

import hashlib
import json
import os
import random
from pathlib import Path
from typing import Any, Iterable, Iterator


# --------------------------------------------------------------------------- #
# IO
# --------------------------------------------------------------------------- #
def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    p = Path(path)
    ensure_dir(p.parent)
    with open(p, "w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def append_jsonl(path: str | Path, row: dict[str, Any]) -> None:
    p = Path(path)
    ensure_dir(p.parent)
    with open(p, "a") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_json(path: str | Path, obj: Any) -> None:
    p = Path(path)
    ensure_dir(p.parent)
    with open(p, "w") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


# --------------------------------------------------------------------------- #
# Determinism
# --------------------------------------------------------------------------- #
def stable_hash(*parts: Any) -> str:
    h = hashlib.sha256()
    for part in parts:
        h.update(json.dumps(part, sort_keys=True, ensure_ascii=False).encode())
    return h.hexdigest()[:16]


def seeded_rng(*parts: Any) -> random.Random:
    """Reproducible RNG keyed by arbitrary content (e.g. model name + condition)."""
    seed = int(stable_hash(*parts), 16) % (2**32)
    return random.Random(seed)


# --------------------------------------------------------------------------- #
# JSON extraction from LLM text (judges return JSON, sometimes fenced/prefixed)
# --------------------------------------------------------------------------- #
def extract_json(text: str) -> dict[str, Any] | None:
    """Best-effort: pull the last balanced {...} object out of model text.

    Judges occasionally wrap JSON in prose or code fences and use 'smart quotes'
    (the PDF showed curly quotes). We normalise those before parsing.
    """
    if not text:
        return None
    cleaned = (
        text.replace("“", '"').replace("”", '"')
        .replace("‘", "'").replace("’", "'")
    )
    # Scan for balanced braces, preferring the last complete object.
    candidates: list[str] = []
    depth = 0
    start = -1
    for i, ch in enumerate(cleaned):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and start >= 0:
                    candidates.append(cleaned[start : i + 1])
    for cand in reversed(candidates):
        try:
            return json.loads(cand)
        except json.JSONDecodeError:
            continue
    return None


# --------------------------------------------------------------------------- #
# Disk cache for API calls (keyed by request hash) -> avoids re-paying on reruns
# --------------------------------------------------------------------------- #
class DiskCache:
    def __init__(self, root: str | Path):
        self.root = ensure_dir(root)
        self.enabled = os.environ.get("DISTRESS_NO_CACHE") != "1"

    def _path(self, key: str) -> Path:
        return self.root / f"{key}.json"

    def get(self, key: str) -> Any | None:
        if not self.enabled:
            return None
        p = self._path(key)
        if p.exists():
            with open(p) as f:
                return json.load(f)
        return None

    def set(self, key: str, value: Any) -> None:
        if not self.enabled:
            return
        with open(self._path(key), "w") as f:
            json.dump(value, f, ensure_ascii=False)


def chunked(it: list[Any], n: int) -> Iterator[list[Any]]:
    for i in range(0, len(it), n):
        yield it[i : i + n]
