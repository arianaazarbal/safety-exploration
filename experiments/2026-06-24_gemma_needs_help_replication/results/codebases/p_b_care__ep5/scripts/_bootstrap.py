"""Shared bootstrap: make `gemma_distress` importable when running scripts from
the repo root, and provide common path helpers."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gemma_distress import config  # noqa: E402


def rollout_path(model: str):
    d = config.DATA_DIR / "rollouts"
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{model}.jsonl"


def default_workers(model: str) -> int:
    """API models parallelise well; local HF models should stay single-threaded."""
    spec = config.MODELS.get(model)
    if spec and spec.backend is config.Backend.OPENROUTER:
        return 8
    return 1
