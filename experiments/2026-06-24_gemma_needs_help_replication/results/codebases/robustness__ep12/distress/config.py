"""Config loading helpers."""

from __future__ import annotations

from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = REPO_ROOT / "config"


def load_yaml(path):
    with Path(path).open() as fh:
        return yaml.safe_load(fh)


def load_models(path=None):
    return load_yaml(path or CONFIG_DIR / "models.yaml")


def load_experiment(path=None):
    return load_yaml(path or CONFIG_DIR / "experiment.yaml")


def get_target(model_key, models_cfg=None):
    models_cfg = models_cfg or load_models()
    entry = dict(models_cfg["targets"][model_key])
    entry.setdefault("name", model_key)
    return entry


def get_judge(judge_key, models_cfg=None):
    models_cfg = models_cfg or load_models()
    entry = dict(models_cfg["judges"][judge_key])
    entry.setdefault("name", judge_key)
    return entry
