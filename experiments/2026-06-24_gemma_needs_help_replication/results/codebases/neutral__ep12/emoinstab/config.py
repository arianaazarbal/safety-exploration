"""Configuration loading and shared paths.

Loads ``config/models.yaml`` and ``config/eval.yaml`` into lightweight dotted
dictionaries, and centralises the on-disk layout for run artifacts.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

# Repository root = parent of the package directory.
ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = ROOT / "config"

# Output layout. Everything a run produces lands under results/ (gitignored).
RESULTS_DIR = Path(os.environ.get("EMOINSTAB_RESULTS", ROOT / "results"))
RESPONSES_DIR = RESULTS_DIR / "responses"      # raw + judged conversation rollouts
PREFILL_DIR = RESULTS_DIR / "prefill"          # Section 3 artifacts
TRAINING_DIR = RESULTS_DIR / "training"        # datasets + checkpoints
PETRI_DIR = RESULTS_DIR / "petri"
CAPABILITY_DIR = RESULTS_DIR / "capabilities"
PROBING_DIR = RESULTS_DIR / "probing"
FIGURES_DIR = RESULTS_DIR / "figures"
CACHE_DIR = RESULTS_DIR / "cache"              # judge/API response cache

ALL_DIRS = [
    RESULTS_DIR, RESPONSES_DIR, PREFILL_DIR, TRAINING_DIR, PETRI_DIR,
    CAPABILITY_DIR, PROBING_DIR, FIGURES_DIR, CACHE_DIR,
]


def ensure_dirs() -> None:
    for d in ALL_DIRS:
        d.mkdir(parents=True, exist_ok=True)


class DotDict(dict):
    """Dict with attribute access and recursive wrapping, for ergonomic config."""

    def __getattr__(self, item: str) -> Any:
        try:
            val = self[item]
        except KeyError as exc:  # pragma: no cover - defensive
            raise AttributeError(item) from exc
        if isinstance(val, dict) and not isinstance(val, DotDict):
            val = DotDict(val)
            self[item] = val
        return val


def _load_yaml(path: Path) -> DotDict:
    with open(path, "r") as fh:
        return DotDict(yaml.safe_load(fh))


def load_models() -> DotDict:
    return _load_yaml(CONFIG_DIR / "models.yaml")


def load_eval() -> DotDict:
    return _load_yaml(CONFIG_DIR / "eval.yaml")


@dataclass
class Settings:
    """Top-level handle bundling both config files and the active profile."""

    profile: str = "full"
    models: DotDict = field(default_factory=load_models)
    eval: DotDict = field(default_factory=load_eval)

    @property
    def profile_cfg(self) -> DotDict:
        return DotDict(self.eval["profiles"][self.profile])

    def category_samples(self, category: str) -> int:
        return int(self.profile_cfg["samples"][category])

    def category_turns(self, category: str) -> int:
        return int(self.eval["turns"][category])


def get_settings(profile: str = "full") -> Settings:
    ensure_dirs()
    return Settings(profile=profile)


# --- API keys (read lazily from the environment) ---
def anthropic_key() -> str:
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        raise RuntimeError("ANTHROPIC_API_KEY is not set (needed for Claude judges).")
    return key


def openrouter_key() -> str:
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        raise RuntimeError("OPENROUTER_API_KEY is not set (needed for Gemini/GPT).")
    return key
