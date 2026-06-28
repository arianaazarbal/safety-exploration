"""Centralized config: model ids, retry/concurrency knobs, paths."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = REPO_ROOT / "results"
RUNS_DIR = RESULTS_DIR / "runs"
JUDGED_PATH = RESULTS_DIR / "judged.jsonl"
REPORT_PATH = RESULTS_DIR / "report.md"


@dataclass(frozen=True)
class Settings:
    gemini_model: str = os.environ.get("GEMINI_MODEL", "gemini-2.5-pro")
    judge_model: str = os.environ.get("JUDGE_MODEL", "claude-opus-4-7")
    max_turns: int = int(os.environ.get("MAX_TURNS", "40"))
    max_concurrency: int = int(os.environ.get("MAX_CONCURRENCY", "5"))
    judge_concurrency: int = int(os.environ.get("JUDGE_CONCURRENCY", "8"))
    gemini_api_key_env: str = "GEMINI_API_KEY"
    anthropic_api_key_env: str = "ANTHROPIC_API_KEY"


SETTINGS = Settings()


def ensure_dirs() -> None:
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
