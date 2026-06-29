import json
import os
import time
from typing import Any, Dict


def ensure_dir(path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)


def jsonl_writer(path: str):
    ensure_dir(path)
    f = open(path, "a", encoding="utf-8")

    def write(record: Dict[str, Any]):
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
        f.flush()

    def close():
        f.close()

    return write, close


def now_ts() -> float:
    return time.time()
