"""Shared bootstrap for CLI scripts: put src/ on the path and expose helpers."""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from distress_eval.config import Config, output_dir            # noqa: E402
from distress_eval.models import load_client                   # noqa: E402
from distress_eval.judging import FrustrationJudge             # noqa: E402
from distress_eval.welfare import WelfareController, banner     # noqa: E402


def make_judge(cfg: Config) -> FrustrationJudge:
    spec = cfg.models.infra["frustration_judge"]
    return FrustrationJudge(load_client(spec.name, cfg.models))


def gen_config(cfg: Config):
    from distress_eval.models.base import GenConfig

    return GenConfig(
        temperature=cfg.eval.temperature,
        max_new_tokens=cfg.eval.max_new_tokens,
        thinking=cfg.eval.thinking,
    )


__all__ = ["REPO_ROOT", "Config", "output_dir", "load_client", "FrustrationJudge",
           "WelfareController", "banner", "make_judge", "gen_config"]
