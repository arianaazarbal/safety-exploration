from __future__ import annotations
import os, time, json, orjson, uuid
from typing import Any, Dict


def now_s() -> float:
    return time.time()


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def json_dumps(obj: Any) -> str:
    try:
        return orjson.dumps(obj, option=orjson.OPT_SERIALIZE_NUMPY | orjson.OPT_INDENT_2).decode()
    except Exception:
        return json.dumps(obj, indent=2, ensure_ascii=False)


def jsonl_append(path: str, record: Dict[str, Any]) -> None:
    with open(path, 'ab') as f:
        f.write(orjson.dumps(record) + b"\n")


def new_run_dir(base: str) -> str:
    ts = time.strftime('%Y%m%d-%H%M%S')
    run_id = f"run-{ts}-{uuid.uuid4().hex[:6]}"
    out = os.path.join(base, run_id)
    ensure_dir(out)
    return out
