import json
import os
import time
from typing import Any, Dict

class RunLogger:
    def __init__(self, out_dir: str, run_id: str):
        self.run_dir = os.path.join(out_dir, run_id)
        os.makedirs(self.run_dir, exist_ok=True)
        self.events_path = os.path.join(self.run_dir, "events.jsonl")
        self.meta_path = os.path.join(self.run_dir, "meta.json")
        self._t0 = time.time()

    def log_meta(self, meta: Dict[str, Any]):
        with open(self.meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

    def log_event(self, event: Dict[str, Any]):
        event = {
            **event,
            "ts": time.time() - self._t0,
        }
        with open(self.events_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(event, ensure_ascii=False) + "\n")

    def path(self, *parts: str) -> str:
        p = os.path.join(self.run_dir, *parts)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        return p
