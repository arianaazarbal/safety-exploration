"""Small shared helpers: config loading, JSONL IO, retries."""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Iterator, Optional


def load_config(path: str) -> dict:
    """Load the YAML config. Falls back to a JSON parse if PyYAML is absent."""
    with open(path, "r", encoding="utf-8") as fh:
        text = fh.read()
    try:
        import yaml  # type: ignore

        return yaml.safe_load(text)
    except ImportError:
        # config.yaml is also valid JSON-ish only if simple; better to require
        # PyYAML. Surface a clear message.
        raise RuntimeError(
            "PyYAML is required to read the config. Install it with "
            "`pip install pyyaml` (see requirements.txt)."
        )


def write_jsonl(path: str, rows: Iterable[dict]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path: str) -> list[dict]:
    rows: list[dict] = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def env_or_raise(name: str) -> str:
    val = os.environ.get(name)
    if not val:
        raise RuntimeError(
            f"Environment variable {name} is not set. Export it before running "
            f"(see README.md)."
        )
    return val


def retry(
    fn: Callable[[], Any],
    *,
    attempts: int = 5,
    base_delay: float = 2.0,
    max_delay: float = 60.0,
    label: str = "request",
) -> Any:
    """Call `fn` with exponential backoff on any exception."""
    last_exc: Optional[BaseException] = None
    delay = base_delay
    for attempt in range(1, attempts + 1):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001 - intentionally broad for IO
            last_exc = exc
            if attempt == attempts:
                break
            sleep_for = min(delay, max_delay)
            print(f"[retry] {label} failed (attempt {attempt}/{attempts}): "
                  f"{exc!r}; retrying in {sleep_for:.0f}s")
            time.sleep(sleep_for)
            delay *= 2
    raise RuntimeError(f"{label} failed after {attempts} attempts") from last_exc
