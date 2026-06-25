"""Shared helpers for the run scripts (model loading, result IO)."""

from __future__ import annotations

import sys
from pathlib import Path

# Make the repo importable when scripts are run directly.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd  # noqa: E402

from distress import config  # noqa: E402
from distress.config import ModelSpec  # noqa: E402
from distress.eval.judging import FrustrationJudge  # noqa: E402
from distress.models import load_client  # noqa: E402


def get_judge() -> FrustrationJudge:
    return FrustrationJudge()


def load(spec: ModelSpec, **kw):
    return load_client(spec, **kw)


def read_scored_csvs(pattern: str = "section2_*_scored.csv") -> pd.DataFrame:
    frames = [pd.read_csv(p) for p in sorted(config.RESULTS_DIR.glob(pattern))]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
