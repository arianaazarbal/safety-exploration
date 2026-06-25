"""Shared CLI bootstrap: makes the repo importable, loads config + logging +
registry. Imported by every script."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make the repo root importable when scripts are run directly.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from eilm.config import load_config  # noqa: E402
from eilm.models.registry import ModelRegistry  # noqa: E402
from eilm.utils.logconf import setup_logging  # noqa: E402


def common_parser(description: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--config", default=None, help="Path to config.yaml")
    p.add_argument("--no-vllm", action="store_true",
                   help="Force the transformers backend for local models")
    return p


def boot(args):
    cfg = load_config(args.config)
    logger = setup_logging(cfg.path("logs"), level=cfg["runtime"]["log_level"],
                           run_name=Path(sys.argv[0]).stem)
    registry = ModelRegistry(cfg, prefer_vllm=not getattr(args, "no_vllm", False))
    return cfg, registry, logger
