"""Shared helpers for the CLI scripts: config loading and client construction.

Scripts are intentionally thin wrappers over the library so that the logic
under test lives in importable, reviewable modules rather than in argv-parsing
glue.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make the package importable when scripts are run directly.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from emotional_instability.config import Config, load_config  # noqa: E402
from emotional_instability.eval.judge import FrustrationJudge  # noqa: E402
from emotional_instability.models.registry import (  # noqa: E402
    build_infra_client,
    build_target_client,
)
from emotional_instability.utils.seeding import seed_everything  # noqa: E402

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def setup(seed: int | None = None) -> Config:
    cfg = load_config()
    seed_everything(seed if seed is not None else cfg.seed)
    DATA_DIR.mkdir(exist_ok=True)
    return cfg


def make_judge(cfg: Config, infra_key: str = "judge") -> FrustrationJudge:
    return FrustrationJudge(build_infra_client(cfg.infra(infra_key)))


def make_target(cfg: Config, model_key: str, *, adapter_path: str | None = None, **kw):
    return build_target_client(cfg.target(model_key), adapter_path=adapter_path, **kw)
