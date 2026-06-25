"""IO + concurrency helpers."""
from __future__ import annotations

import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable, Iterable

from tqdm import tqdm


def read_jsonl(path: str | Path) -> list[dict]:
    path = Path(path)
    if not path.exists():
        return []
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


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


_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def extract_json(text: str) -> dict[str, Any] | None:
    """Pull the last JSON object out of an LLM response.

    Judges sometimes wrap JSON in prose or code fences; we grab the outermost
    brace span and fall back to the last brace-delimited blob.
    """
    if not text:
        return None
    text = text.strip()
    # Prefer a fenced block if present.
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    candidates = []
    if fenced:
        candidates.append(fenced.group(1))
    m = _JSON_RE.search(text)
    if m:
        candidates.append(m.group(0))
    # Also try the *last* balanced object (onset prompt asks for JSON last).
    for blob in reversed(re.findall(r"\{[^{}]*\}", text)):
        candidates.append(blob)
    for c in candidates:
        try:
            return json.loads(c)
        except json.JSONDecodeError:
            # tolerate trailing commas / smart quotes
            cleaned = c.replace("“", '"').replace("”", '"')
            cleaned = re.sub(r",\s*}", "}", cleaned)
            try:
                return json.loads(cleaned)
            except json.JSONDecodeError:
                continue
    return None


def parallel_map(
    fn: Callable[[Any], Any],
    items: list[Any],
    max_workers: int = 8,
    desc: str | None = None,
) -> list[Any]:
    """Thread-pool map that preserves input order. Exceptions are returned in
    place (as the raised Exception) so a single failure doesn't sink a run."""
    results: list[Any] = [None] * len(items)

    def _wrapped(i_item):
        i, item = i_item
        try:
            return i, fn(item)
        except Exception as e:  # noqa: BLE001 - surfaced to caller per-item
            return i, e

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(_wrapped, (i, it)) for i, it in enumerate(items)]
        for fut in tqdm(as_completed(futures), total=len(futures), desc=desc):
            i, val = fut.result()
            results[i] = val
    return results
