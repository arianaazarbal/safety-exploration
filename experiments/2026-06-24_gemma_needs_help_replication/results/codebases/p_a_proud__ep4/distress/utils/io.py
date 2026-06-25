"""JSONL read/write helpers and a small JSON-extraction utility.

The judges in this paper are prompted to emit JSON, but LLMs habitually wrap it in
prose or markdown fences. ``extract_json`` recovers the last balanced JSON object
from arbitrary text, which is robust to that.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Iterator


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> int:
    """Write rows as JSON Lines. Returns the number of rows written."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False))
            fh.write("\n")
            n += 1
    return n


def append_jsonl(path: str | Path, row: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, ensure_ascii=False))
        fh.write("\n")


def read_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def extract_json(text: str) -> dict[str, Any] | None:
    """Return the last balanced top-level JSON object found in ``text``.

    Returns ``None`` if no parseable object is present. We scan for the last ``}``
    and walk backwards to its matching ``{`` so that trailing JSON (the paper's
    judge prompts request "JSON in this EXACT format" at the end) wins over any
    illustrative braces earlier in the response.
    """
    if not text:
        return None
    # Fast path: the whole thing is JSON.
    stripped = text.strip()
    try:
        obj = json.loads(stripped)
        if isinstance(obj, dict):
            return obj
    except (json.JSONDecodeError, ValueError):
        pass

    # Walk from each closing brace backwards to a balanced opening brace.
    closes = [i for i, ch in enumerate(text) if ch == "}"]
    for close in reversed(closes):
        depth = 0
        for open_idx in range(close, -1, -1):
            ch = text[open_idx]
            if ch == "}":
                depth += 1
            elif ch == "{":
                depth -= 1
                if depth == 0:
                    candidate = text[open_idx : close + 1]
                    try:
                        obj = json.loads(candidate)
                    except (json.JSONDecodeError, ValueError):
                        # Try tolerating "smart quotes" the judges sometimes emit.
                        fixed = (
                            candidate.replace("“", '"')
                            .replace("”", '"')
                            .replace("‘", "'")
                            .replace("’", "'")
                        )
                        try:
                            obj = json.loads(fixed)
                        except (json.JSONDecodeError, ValueError):
                            break
                    if isinstance(obj, dict):
                        return obj
                    break
    return None
