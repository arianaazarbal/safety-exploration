"""Shared utilities: IO (jsonl with resume), retries, JSON extraction, logging."""
from __future__ import annotations

import json
import logging
import os
import re
import time
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, TypeVar

logging.basicConfig(
    level=os.environ.get("GD_LOGLEVEL", "INFO"),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


log = get_logger(__name__)

T = TypeVar("T")


# --------------------------------------------------------------------------- IO
def _default(o: Any) -> Any:
    if is_dataclass(o) and not isinstance(o, type):
        return asdict(o)
    raise TypeError(f"Not JSON serializable: {type(o)}")


def write_jsonl(path: str | Path, rows: Iterable[Any], mode: str = "w") -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open(mode, encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, default=_default, ensure_ascii=False) + "\n")


def append_jsonl(path: str | Path, row: Any) -> None:
    write_jsonl(path, [row], mode="a")


def read_jsonl(path: str | Path) -> list[dict]:
    path = Path(path)
    if not path.exists():
        return []
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def completed_ids(path: str | Path, key: str = "id") -> set:
    """IDs already present in an output jsonl, for resuming interrupted runs."""
    return {r[key] for r in read_jsonl(path) if key in r}


# ------------------------------------------------------------------- JSON parse
_JSON_OBJ = re.compile(r"\{.*\}", re.DOTALL)


def extract_json(text: str) -> dict | None:
    """Extract the last JSON object from possibly-chatty model output.

    Handles fancy quotes the judge sometimes emits and trailing prose.
    """
    if not text:
        return None
    # Normalise curly quotes that break json.loads.
    cleaned = (
        text.replace("“", '"').replace("”", '"')
        .replace("‘", "'").replace("’", "'")
    )
    # Try whole string first, then the widest brace span, then per-match.
    candidates = []
    m = _JSON_OBJ.search(cleaned)
    if m:
        candidates.append(m.group(0))
    candidates += re.findall(r"\{[^{}]*\}", cleaned)
    for cand in candidates:
        try:
            return json.loads(cand)
        except json.JSONDecodeError:
            continue
    return None


# ------------------------------------------------------------------ retry/throttle
def retry(
    fn: Callable[[], T],
    *,
    attempts: int = 6,
    base_delay: float = 2.0,
    exceptions: tuple = (Exception,),
    sleep: Callable[[float], None] = time.sleep,
) -> T:
    """Exponential-backoff retry, for flaky API calls."""
    last = None
    for i in range(attempts):
        try:
            return fn()
        except exceptions as e:  # noqa: BLE001
            last = e
            delay = base_delay * (2**i)
            log.warning("attempt %d/%d failed: %s (retrying in %.1fs)", i + 1, attempts, e, delay)
            sleep(delay)
    raise RuntimeError(f"all {attempts} attempts failed") from last


def batched(it: Iterable[T], n: int) -> Iterator[list[T]]:
    batch: list[T] = []
    for x in it:
        batch.append(x)
        if len(batch) >= n:
            yield batch
            batch = []
    if batch:
        yield batch


def data_dir() -> Path:
    return Path(os.environ.get("GD_DATA_DIR", "data"))
