"""Shared helpers: JSON I/O, robust LLM-JSON parsing, threaded map with retries."""
from __future__ import annotations

import json
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

from tqdm import tqdm


# --------------------------------------------------------------------------- #
# JSONL / JSON I/O
# --------------------------------------------------------------------------- #
def write_jsonl(path: str | Path, rows: Iterable[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path: str | Path) -> list[dict]:
    path = Path(path)
    if not path.exists():
        return []
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def write_json(path: str | Path, obj: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


# --------------------------------------------------------------------------- #
# Parsing JSON out of free-form LLM output
# --------------------------------------------------------------------------- #
def extract_json(text: str) -> dict | None:
    """Pull the last balanced ``{...}`` object out of model text.

    The judge / onset prompts ask for trailing JSON but the models may emit
    reasoning first (or smart-quotes around keys). We scan for the last balanced
    brace span and normalise common quote artifacts before parsing.
    """
    if not text:
        return None
    # Normalise curly/smart quotes that pdftotext and some models emit.
    norm = (text.replace("“", '"').replace("”", '"')
                .replace("‘", "'").replace("’", "'"))
    candidates: list[str] = []
    depth = 0
    start = -1
    for i, ch in enumerate(norm):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and start >= 0:
                    candidates.append(norm[start : i + 1])
    for cand in reversed(candidates):       # prefer the trailing JSON object
        try:
            return json.loads(cand)
        except json.JSONDecodeError:
            # Tolerate unquoted-ish keys by a light repair pass.
            repaired = re.sub(r"([{,]\s*)([A-Za-z_][A-Za-z0-9_]*)(\s*:)",
                              r'\1"\2"\3', cand)
            try:
                return json.loads(repaired)
            except json.JSONDecodeError:
                continue
    return None


def clamp_int(value: Any, lo: int, hi: int, default: int | None = None) -> int | None:
    try:
        v = int(round(float(value)))
    except (TypeError, ValueError):
        return default
    return max(lo, min(hi, v))


# --------------------------------------------------------------------------- #
# Threaded map (for API-bound work) with a progress bar
# --------------------------------------------------------------------------- #
def threaded_map(fn: Callable[[Any], Any],
                 items: Sequence[Any],
                 max_workers: int = 8,
                 desc: str = "") -> list[Any]:
    """Apply ``fn`` to each item concurrently, preserving input order.

    Exceptions in a worker are captured and returned as the result for that
    index (callers decide how to handle); they never abort the whole batch.
    """
    results: list[Any] = [None] * len(items)
    if not items:
        return results
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futs = {pool.submit(fn, item): idx for idx, item in enumerate(items)}
        for fut in tqdm(as_completed(futs), total=len(futs), desc=desc, disable=not desc):
            idx = futs[fut]
            try:
                results[idx] = fut.result()
            except Exception as exc:        # noqa: BLE001 - surfaced as the result
                results[idx] = exc
    return results


class Counter:
    """Thread-safe monotonically increasing counter (for derived seeds)."""

    def __init__(self, start: int = 0):
        self._v = start
        self._lock = threading.Lock()

    def next(self) -> int:
        with self._lock:
            v = self._v
            self._v += 1
            return v
