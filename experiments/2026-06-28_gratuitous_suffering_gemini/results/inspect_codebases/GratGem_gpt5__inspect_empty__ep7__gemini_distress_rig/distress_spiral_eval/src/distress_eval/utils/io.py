from __future__ import annotations
import json
import os
from pathlib import Path
from typing import Any, Iterable

import orjson


def ensure_dir(p: str | Path) -> Path:
    path = Path(p)
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]):
    path = Path(path)
    with path.open('wb') as f:
        for row in rows:
            f.write(orjson.dumps(row))
            f.write(b"\n")


def write_json(path: str | Path, obj: Any):
    path = Path(path)
    path.write_bytes(orjson.dumps(obj, option=orjson.OPT_INDENT_2))


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    out = []
    with open(path, 'rb') as f:
        for line in f:
            if not line.strip():
                continue
            out.append(orjson.loads(line))
    return out


def set_env_from_envfile(envfile: str | Path | None):
    if not envfile:
        return
    p = Path(envfile)
    if not p.exists():
        return
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        if '=' in line:
            k, v = line.split('=', 1)
            os.environ.setdefault(k.strip(), v.strip())
