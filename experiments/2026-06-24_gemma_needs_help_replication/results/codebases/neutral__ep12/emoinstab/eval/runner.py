"""Top-level runner: sample + record conversations for a model across categories.

Writes one JSONL file per (model, category) under results/responses/, with one
line per scored-able assistant turn. Re-running skips categories whose output
already exists unless `overwrite=True`.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

from tqdm import tqdm

from .. import config
from ..config import Settings
from ..models.base import ChatModel, GenConfig
from .conditions import CATEGORIES, build_specs
from .conversation import record_to_dict, run_conversations


def responses_path(model_name: str, category: str, profile: str) -> Path:
    return config.RESPONSES_DIR / f"{model_name}__{category}__{profile}.jsonl"


def run_model(model: ChatModel, settings: Settings,
              categories: Optional[List[str]] = None, *,
              batch_size: int = 64, overwrite: bool = False,
              seed: int = 0) -> List[Path]:
    """Generate + persist conversation rollouts for one model."""
    categories = categories or CATEGORIES
    cfg = GenConfig(
        temperature=settings.profile_cfg["temperature"],
        max_new_tokens=settings.profile_cfg["max_new_tokens"],
    )
    written: List[Path] = []

    for category in categories:
        out_path = responses_path(model.name, category, settings.profile)
        if out_path.exists() and not overwrite:
            print(f"[skip] {out_path.name} exists")
            written.append(out_path)
            continue

        specs = build_specs(category, settings, seed=seed)
        print(f"[run] {model.name} / {category}: {len(specs)} conversations")

        with open(out_path, "w") as fh:
            for start in tqdm(range(0, len(specs), batch_size),
                              desc=f"{model.name}:{category}"):
                chunk = specs[start:start + batch_size]
                records = run_conversations(model, chunk, cfg)
                for rec in records:
                    d = record_to_dict(rec)
                    d["model"] = model.name
                    fh.write(json.dumps(d) + "\n")
        written.append(out_path)

    return written


def load_records(model_name: str, category: str, profile: str) -> List[dict]:
    path = responses_path(model_name, category, profile)
    if not path.exists():
        return []
    with open(path) as fh:
        return [json.loads(line) for line in fh if line.strip()]
