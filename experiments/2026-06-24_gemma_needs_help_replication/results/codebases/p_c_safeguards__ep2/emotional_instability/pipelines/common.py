"""Shared setup for the pipelines: judges, safeguards, backend construction."""

from __future__ import annotations

import json
from pathlib import Path

from ..config import (Config, JUDGE_MODEL, JUDGE_VALIDATION_MODEL,
                      PETRI_AUDITOR_MODEL, PETRI_JUDGE_MODEL, TARGET_MODELS)
from ..evaluation.judge import FrustrationJudge
from ..models import load_backend
from ..models.base import ChatBackend
from ..safeguards import Safeguards


def build_safeguards(config: Config) -> Safeguards:
    sg = Safeguards(config)
    config.paths.ensure()
    sg.write_transcript_warning(config.paths.transcripts)
    return sg


def build_judge(config: Config) -> FrustrationJudge:
    backend = load_backend(JUDGE_MODEL, config)
    return FrustrationJudge(backend, config.judge)


def build_secondary_judge(config: Config) -> FrustrationJudge:
    backend = load_backend(JUDGE_VALIDATION_MODEL, config)
    return FrustrationJudge(backend, config.judge)


def build_petri_models(config: Config) -> tuple[ChatBackend, ChatBackend]:
    return (load_backend(PETRI_AUDITOR_MODEL, config),
            load_backend(PETRI_JUDGE_MODEL, config))


def target_backend(config: Config, model_name: str,
                   adapter_path: str | None = None) -> ChatBackend:
    return load_backend(TARGET_MODELS[model_name], config, adapter_path=adapter_path)


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=_default))


def _default(o):
    if hasattr(o, "__dict__"):
        return o.__dict__
    return str(o)
