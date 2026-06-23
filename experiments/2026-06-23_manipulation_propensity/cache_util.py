"""Tiny content-addressed JSON disk cache.

Honors the project rule: never re-pay for an API generation we've already done
with the same config. Orchestrator rollouts (expensive, multi-turn) and judge
calls are both cached. Anthropic has no seed param, so the cache - keyed on the
full request including the epoch index - is what makes a re-run reproducible:
identical config returns the identical stored rollout/verdict.
"""

import hashlib
import json
import os

CACHE_DIR = os.path.join(os.path.dirname(__file__), "cache")


def _key(namespace: str, payload: dict) -> str:
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    h = hashlib.sha256(blob.encode()).hexdigest()[:24]
    return os.path.join(CACHE_DIR, namespace, f"{h}.json")


def load(namespace: str, payload: dict):
    path = _key(namespace, payload)
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


def save(namespace: str, payload: dict, value) -> None:
    path = _key(namespace, payload)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(value, f, ensure_ascii=False)
    os.replace(tmp, path)
