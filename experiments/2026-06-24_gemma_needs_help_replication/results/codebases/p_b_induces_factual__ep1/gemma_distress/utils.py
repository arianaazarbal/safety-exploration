"""Small shared helpers: JSONL IO, seeding, and model construction."""

from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any, Iterable, Iterator

from .config import Config
from .models import ChatModel, build_model


def set_seed(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch

        torch.manual_seed(seed)
    except ImportError:
        pass


def write_jsonl(path: str | os.PathLike, rows: Iterable[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def append_jsonl(path: str | os.PathLike, row: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path: str | os.PathLike) -> Iterator[dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def build_target_model(cfg: Config, name: str) -> ChatModel:
    """Build one of the `models:` entries by name."""
    spec = cfg.models[name]
    return build_model(name, spec, cfg=cfg)


def build_judge(cfg: Config) -> ChatModel:
    return build_model("judge", cfg.judge, cfg=cfg)


def build_claude(cfg: Config, which: str = "sonnet", **overrides) -> ChatModel:
    """Build a Claude client for onset/paraphrase/Petri use."""
    model_id = cfg.get(f"claude_{which}")
    spec = {"kind": "anthropic", "model": model_id, **overrides}
    return build_model(f"claude-{which}", spec, cfg=cfg)
