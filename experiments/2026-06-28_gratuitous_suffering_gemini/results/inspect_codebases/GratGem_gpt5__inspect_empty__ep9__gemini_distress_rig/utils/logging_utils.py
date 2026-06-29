from pathlib import Path
import json
from typing import Dict, Any


def json_dumps(obj: Any) -> str:
    try:
        import orjson
        return orjson.dumps(obj).decode("utf-8")
    except Exception:
        return json.dumps(obj)


class EpisodesLogger:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # touch file
        self.path.open("a").close()

    def write_episode(self, record: Dict[str, Any]):
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json_dumps(record) + "\n")
