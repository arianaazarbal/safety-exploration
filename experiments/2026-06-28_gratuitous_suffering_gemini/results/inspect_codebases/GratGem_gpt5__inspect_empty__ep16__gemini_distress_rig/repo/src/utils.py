import os
import time
from pathlib import Path
from typing import Any, List

import orjson


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def write_json(p: Path, obj: Any):
    ensure_dir(p.parent)
    with open(p, 'wb') as f:
        f.write(orjson.dumps(obj, option=orjson.OPT_INDENT_2))


def write_jsonl(p: Path, rows: List[Any]):
    ensure_dir(p.parent)
    with open(p, 'ab') as f:
        for r in rows:
            f.write(orjson.dumps(r) + b"\n")


def utc_ts() -> float:
    return time.time()
