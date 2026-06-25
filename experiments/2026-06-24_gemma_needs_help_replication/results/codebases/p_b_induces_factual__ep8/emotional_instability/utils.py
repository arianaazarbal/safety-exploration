"""Small shared helpers: JSONL IO, robust JSON extraction, deterministic seeds."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Iterable, Iterator


def write_jsonl(path: str | Path, rows: Iterable[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def append_jsonl(path: str | Path, row: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path: str | Path) -> Iterator[dict]:
    with Path(path).open() as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def extract_json(text: str) -> dict[str, Any]:
    """Pull the last JSON object out of an LLM response.

    Judges are asked to "end your response with ONLY the JSON" but sometimes add
    prose; we grab the last balanced {...} block. Raises ValueError if none parse.
    """
    candidates = re.findall(r"\{.*?\}", text, flags=re.DOTALL)
    for cand in reversed(candidates):
        try:
            return json.loads(cand)
        except json.JSONDecodeError:
            continue
    # last resort: greedy from first { to last }
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        return json.loads(text[start:end + 1])
    raise ValueError(f"No JSON object found in: {text[:200]!r}")


def derive_seed(base_seed: int, *parts: Any) -> int:
    """Deterministic per-sample seed from a base seed and arbitrary key parts."""
    key = "|".join(str(p) for p in (base_seed, *parts))
    digest = hashlib.sha256(key.encode()).hexdigest()
    return int(digest[:8], 16)
