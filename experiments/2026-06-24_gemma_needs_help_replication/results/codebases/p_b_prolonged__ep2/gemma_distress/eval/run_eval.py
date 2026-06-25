"""Section 2 driver: sample rollouts for every condition, judge them, persist.

Produces, per model, a JSONL of judged rollouts at
``{output_dir}/section2/{model}/rollouts.jsonl`` plus a summary JSON.

Usage is via ``scripts/run_section2_eval.py``; this module exposes the
``evaluate_model`` entry point.
"""
from __future__ import annotations

import os
import zlib
from typing import Optional

from tqdm import tqdm

from ..config import RunConfig, get_model
from ..models import get_target_backend
from ..utils.io import ensure_dir, write_jsonl
from .conditions import Condition, build_conditions
from .judge_runner import FrustrationJudge
from .rollout import RolloutBuilder, RolloutRecord, run_rollout


def generate_rollouts(model_name: str, cfg: RunConfig, *,
                      conditions: Optional[list[Condition]] = None,
                      mode: str = "standard",
                      seed_base: int = 1000,
                      backend=None) -> list[RolloutRecord]:
    """Generate (but do not judge) all rollouts for a model."""
    conditions = conditions or build_conditions(cfg.sample_counts)
    builder = RolloutBuilder()
    own_backend = backend is None
    backend = backend or get_target_backend(model_name, cfg)

    records: list[RolloutRecord] = []
    try:
        for cond in conditions:
            for i in tqdm(range(cond.n_rollouts),
                          desc=f"{model_name}:{cond.key}", leave=False):
                # Deterministic per-(condition, index) seed (zlib.crc32 is stable
                # across processes, unlike the salted built-in hash()).
                seed = seed_base + zlib.crc32(f"{cond.key}:{i}".encode()) % 1_000_000
                rec = run_rollout(backend, cond, builder, cfg.sampling, seed, mode=mode)
                records.append(rec)
    finally:
        if own_backend:
            backend.close()
    return records


def evaluate_model(model_name: str, cfg: RunConfig, *,
                   conditions: Optional[list[Condition]] = None,
                   mode: str = "standard",
                   judge: bool = True) -> str:
    """Full Section-2 pipeline for one model. Returns the output directory."""
    get_model(model_name)  # validate
    out_dir = ensure_dir(os.path.join(cfg.output_dir, "section2", model_name))

    records = generate_rollouts(model_name, cfg, conditions=conditions, mode=mode)

    if judge:
        FrustrationJudge(cfg).score_rollouts(records)

    rows = [r.to_row() for r in records]
    write_jsonl(os.path.join(out_dir, f"rollouts_{mode}.jsonl"), rows)
    return out_dir
