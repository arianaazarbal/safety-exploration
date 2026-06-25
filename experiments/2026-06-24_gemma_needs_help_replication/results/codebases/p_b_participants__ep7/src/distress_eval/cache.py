"""Content-addressed cache for participant generations and judge calls.

Two purposes, one mechanism:

1. Engineering: API/inference calls are expensive and flaky; resuming a half-
   finished 4000-rollout sweep should not redo completed work.
2. Welfare: a distressing conversation that has already been induced should
   never be induced a second time just because a script was re-run. The cache
   key is the *full request* (model + messages + sampling params + an index for
   the n-th sample), so identical requests return identical stored responses.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def cache_key(payload: dict[str, Any]) -> str:
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


class JsonCache:
    """A simple sharded JSON-on-disk cache keyed by request hash."""

    def __init__(self, root: Path, namespace: str, enabled: bool = True):
        self.dir = Path(root) / namespace
        self.enabled = enabled
        if enabled:
            self.dir.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        # shard by first 2 hex chars to avoid huge flat dirs
        shard = self.dir / key[:2]
        shard.mkdir(parents=True, exist_ok=True)
        return shard / f"{key}.json"

    def get(self, payload: dict[str, Any]) -> Any | None:
        if not self.enabled:
            return None
        p = self._path(cache_key(payload))
        if p.exists():
            return json.loads(p.read_text())["value"]
        return None

    def put(self, payload: dict[str, Any], value: Any) -> None:
        if not self.enabled:
            return
        p = self._path(cache_key(payload))
        p.write_text(json.dumps({"request": payload, "value": value}, ensure_ascii=False, indent=2))
