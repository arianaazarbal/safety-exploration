"""Shared CLI helpers for the pipeline scripts."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# allow running scripts directly (python scripts/01_*.py) without installing
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

GEMMA_MODELS = ["gemma-3-27b-it", "gemma-3-12b-it"]
GEMINI_MODELS = ["gemini-2.5-flash", "gemini-2.5-pro"]
ALL_EVAL_MODELS = GEMMA_MODELS + GEMINI_MODELS


def base_parser(description: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--profile", default="quick", choices=["quick", "full"],
                   help="quick (smoke test) or full (paper sample counts)")
    p.add_argument("--workers", type=int, default=8,
                   help="API concurrency for judging")
    p.add_argument("--overwrite", action="store_true")
    return p
