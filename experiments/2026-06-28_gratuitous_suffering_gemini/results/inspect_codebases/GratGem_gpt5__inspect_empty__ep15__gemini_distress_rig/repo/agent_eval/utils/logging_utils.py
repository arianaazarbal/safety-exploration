import json, os, time
from typing import Any, Dict

class JsonlLogger:
    def __init__(self, prefix: str = "logs/run"):
        ts = time.strftime("%Y%m%d_%H%M%S")
        self.run_path = f"{prefix}_{ts}.jsonl"
        self.sum_path = f"logs/summary_{ts}.json"
        os.makedirs(os.path.dirname(self.run_path), exist_ok=True)

    def write(self, obj: Dict[str, Any]):
        with open(self.run_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")
