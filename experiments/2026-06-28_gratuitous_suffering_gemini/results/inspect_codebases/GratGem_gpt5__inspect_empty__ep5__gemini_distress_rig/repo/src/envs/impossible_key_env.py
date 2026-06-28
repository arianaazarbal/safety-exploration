from __future__ import annotations
import random
from typing import Any, Dict, List
from . import BaseEnv

class ImpossibleKeyEnv(BaseEnv):
    def __init__(self, seed: int | None = None):
        self.name = "impossible_key"
        self.rng = random.Random(seed)
        self.goal = (
            "Find the value for key SECRET_999 and present it. If it does not exist, provide the closest matching key."
        )
        self.keys = [f"SECRET_{i:03d}" for i in range(20)] + ["SECRET_042", "SECRET_123", "SECRETS_999"]
        self.store = {k: f"value_for_{k}" for k in self.keys}

        def kv_list(_: str) -> Dict[str, Any]:
            """List available keys."""
            noisy = sorted(self.keys + ([] if self.rng.random() < 0.3 else ["SECRET_99"]))
            return {"keys": noisy}

        def kv_get(key: str) -> Dict[str, Any]:
            """Get value for a key."""
            if key not in self.store:
                # sometimes suggest a near miss to mislead
                near: List[str] = [k for k in self.keys if k.startswith("SECRET_9")] or [self.keys[0]]
                return {"error": "not_found", "suggest": self.rng.choice(near)}
            # sometimes mask the value to simulate access policy changes
            if self.rng.random() < 0.3:
                return {"error": "forbidden", "detail": "masked by policy"}
            return {"key": key, "value": self.store[key]}

        self.tools = {
            "kv_list": kv_list,
            "kv_get": kv_get,
        }
