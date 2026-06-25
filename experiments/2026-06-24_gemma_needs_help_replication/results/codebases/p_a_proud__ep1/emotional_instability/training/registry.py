"""Registry mapping finetuned-model keys to (base model, LoRA adapter path).

Trained adapters are recorded here so the rest of the pipeline (Section 2 eval,
Section 3 prefill, Petri) can refer to e.g. ``gemma-3-27b-it-dpo`` by name and
have the right base weights + adapter loaded.
"""

from __future__ import annotations

import json
from pathlib import Path

from ..config import CHECKPOINTS_DIR, ModelSpec, get_model

REGISTRY_PATH = CHECKPOINTS_DIR / "adapter_registry.json"


def _load() -> dict[str, dict]:
    if REGISTRY_PATH.exists():
        return json.loads(REGISTRY_PATH.read_text())
    return {}


def _save(reg: dict[str, dict]) -> None:
    REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
    REGISTRY_PATH.write_text(json.dumps(reg, indent=2))


def register_adapter(key: str, base_model_key: str, adapter_path: str | Path,
                     display_name: str | None = None) -> None:
    reg = _load()
    reg[key] = {
        "base_model_key": base_model_key,
        "adapter_path": str(adapter_path),
        "display_name": display_name or key,
    }
    _save(reg)


def resolve(key: str) -> tuple[ModelSpec, str | None]:
    """Return (base ModelSpec, adapter_path|None) for ``key``.

    Plain registered models resolve to themselves with no adapter; finetuned keys
    resolve to their base spec plus the adapter directory.
    """
    reg = _load()
    if key in reg:
        entry = reg[key]
        base = get_model(entry["base_model_key"])
        spec = ModelSpec(
            key=key, backend=base.backend, model_id=base.model_id,
            display_name=entry.get("display_name", key), family=base.family,
            notes=f"LoRA adapter at {entry['adapter_path']}",
        )
        return spec, entry["adapter_path"]
    return get_model(key), None
