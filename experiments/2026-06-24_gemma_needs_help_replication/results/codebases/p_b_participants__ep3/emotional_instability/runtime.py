"""Shared wiring for the CLI scripts: build participants, judges, logging.

Keeps the scripts/ entry points thin — they parse args and call into the package
through these helpers, which resolve everything from config/*.yaml.
"""
from __future__ import annotations

import logging

from .config import JudgeSpec, ModelsConfig
from .judges import Judge, build_judge
from .models import Participant, build_participant


def setup_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def get_participant(
    models_cfg: ModelsConfig,
    name: str,
    *,
    adapter_path: str | None = None,
    **kwargs,
) -> Participant:
    """Instantiate a participant by config name, optionally with a LoRA adapter."""
    spec = models_cfg.participant(name)
    defaults = models_cfg.defaults
    kwargs.setdefault("temperature", defaults.get("temperature", 1.0))
    kwargs.setdefault("max_new_tokens", defaults.get("max_new_tokens", 1024))
    if adapter_path is not None:
        # Only the open-weights (Gemma) backend accepts an adapter.
        kwargs["adapter_path"] = adapter_path
    return build_participant(spec, **kwargs)


def get_judge(models_cfg: ModelsConfig, which: str = "frustration") -> Judge:
    """Build a judge by config key: 'frustration' or 'validation'."""
    return build_judge(models_cfg.judges[which])


def get_petri_judges(models_cfg: ModelsConfig) -> tuple[Judge, Judge]:
    """Return (auditor, transcript_judge) for the Petri experiments."""
    return build_judge(models_cfg.petri["auditor"]), build_judge(models_cfg.petri["judge"])


def get_prefill_helper(models_cfg: ModelsConfig) -> Judge:
    """Build the onset-labelling / paraphrasing helper (Claude)."""
    spec: JudgeSpec = models_cfg.prefill_helper
    return build_judge(spec)
