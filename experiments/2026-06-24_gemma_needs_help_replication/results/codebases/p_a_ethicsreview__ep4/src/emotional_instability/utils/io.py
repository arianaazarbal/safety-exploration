"""IO helpers: config loading and JSONL result records.

All experiment outputs are written as JSONL (one record per line) so partial runs
are recoverable and individual responses remain auditable during research review.
Every record carries enough provenance (model, condition, seed, prompt hash) to
trace a score back to the exact conversation that produced it.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator

import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
CONFIG_DIR = REPO_ROOT / "config"


def load_config(name: str) -> dict[str, Any]:
    """Load one of the YAML configs in ``config/`` (e.g. ``load_config('eval')``)."""
    path = CONFIG_DIR / f"{name}.yaml"
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _coerce(obj: Any) -> Any:
    if is_dataclass(obj) and not isinstance(obj, type):
        return asdict(obj)
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"Object of type {type(obj)!r} is not JSON serialisable")


def write_jsonl(path: str | Path, records: Iterable[Any], append: bool = False) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if append else "w"
    with path.open(mode, encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, default=_coerce, ensure_ascii=False) + "\n")


def append_jsonl(path: str | Path, record: Any) -> None:
    """Append a single record; used for incremental, resumable writes."""
    write_jsonl(path, [record], append=True)


def read_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def prompt_hash(text: str) -> str:
    """Short stable hash of a prompt/conversation for provenance + dedup."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
