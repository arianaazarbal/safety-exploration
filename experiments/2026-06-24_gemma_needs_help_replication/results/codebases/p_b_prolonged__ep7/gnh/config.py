"""Configuration loading and lightweight typed accessors.

Two YAML files drive everything:
  config/models.yaml       -> model registry + backend selection
  config/experiments.yaml  -> per-experiment hyperparameters and sample sizes

We deliberately keep config as plain dicts (with a few helpers) rather than a
rigid schema, so that overriding sample sizes from the CLI stays trivial.
"""
from __future__ import annotations

import copy
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = REPO_ROOT / "config"
# Default output root; override with GNH_OUTPUT_DIR.
OUTPUT_DIR = Path(os.environ.get("GNH_OUTPUT_DIR", REPO_ROOT / "outputs"))


def _load_yaml(path: Path) -> dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f)


@dataclass
class ModelSpec:
    name: str
    backend: str                      # "hf" | "openrouter"
    family: str
    role: str                         # "target" | "infrastructure"
    hf_id: Optional[str] = None
    api_id: Optional[str] = None
    is_instruct: bool = True
    num_layers: Optional[int] = None
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def supports_logits(self) -> bool:
        return self.backend == "hf"

    @property
    def supports_prefill(self) -> bool:
        # Prefilled assistant continuation requires raw generation control.
        return self.backend == "hf"

    @property
    def trainable(self) -> bool:
        return self.backend == "hf"


class Config:
    """Holds the merged model registry and experiment settings."""

    def __init__(self, models_path: Path | None = None, exp_path: Path | None = None):
        self._models_raw = _load_yaml(models_path or CONFIG_DIR / "models.yaml")
        self.experiments = _load_yaml(exp_path or CONFIG_DIR / "experiments.yaml")
        self.model_defaults = self._models_raw.get("defaults", {})
        self._models: dict[str, ModelSpec] = {}
        for name, spec in self._models_raw["models"].items():
            known = {"backend", "family", "role", "hf_id", "api_id",
                     "is_instruct", "num_layers"}
            self._models[name] = ModelSpec(
                name=name,
                backend=spec["backend"],
                family=spec["family"],
                role=spec.get("role", "target"),
                hf_id=spec.get("hf_id"),
                api_id=spec.get("api_id"),
                is_instruct=spec.get("is_instruct", True),
                num_layers=spec.get("num_layers"),
                extra={k: v for k, v in spec.items() if k not in known},
            )

    # -- model registry ----------------------------------------------------
    def model(self, name: str) -> ModelSpec:
        if name not in self._models:
            raise KeyError(f"Unknown model '{name}'. Known: {sorted(self._models)}")
        return self._models[name]

    def targets(self) -> list[ModelSpec]:
        return [m for m in self._models.values() if m.role == "target"]

    def register_finetune(self, name: str, base: str) -> ModelSpec:
        """Register an in-repo finetuned checkpoint (e.g. gemma-3-27b-it-dpo).

        Finetunes inherit the base model's backend/family but point hf_id at a
        local adapter directory resolved at load time.
        """
        base_spec = self.model(base)
        spec = ModelSpec(
            name=name, backend=base_spec.backend, family=base_spec.family,
            role="target", hf_id=base_spec.hf_id, is_instruct=True,
            num_layers=base_spec.num_layers,
            extra={"base_model": base, "adapter_dir": str(OUTPUT_DIR / "checkpoints" / name)},
        )
        self._models[name] = spec
        return spec

    # -- experiment config helpers -----------------------------------------
    def section(self, key: str) -> dict[str, Any]:
        return copy.deepcopy(self.experiments[key])

    def apply_profile(self, profile: Optional[str]) -> None:
        """Scale sample sizes down for a smoke run. Mutates self.experiments."""
        if not profile:
            return
        prof = self.experiments.get("profiles", {}).get(profile)
        if prof is None:
            raise KeyError(f"Unknown profile '{profile}'")
        if "section2_scale" in prof:
            scale = prof["section2_scale"]
            for cat in self.experiments["section2"]["categories"].values():
                cat["n_conversations"] = max(1, int(cat["n_conversations"] * scale))
        if "section3_continuations_per_prefill" in prof:
            self.experiments["section3"]["continuations_per_prefill"] = \
                prof["section3_continuations_per_prefill"]
        if "section4_calm_n_target" in prof:
            self.experiments["section4"]["calm_data"]["n_target"] = \
                prof["section4_calm_n_target"]
        if "petri_transcripts_per_emotion" in prof:
            self.experiments["petri"]["transcripts_per_emotion"] = \
                prof["petri_transcripts_per_emotion"]
        if "internal_zscore_calibration_samples" in prof:
            self.experiments["internal_emotion"]["zscore_calibration_samples"] = \
                prof["internal_zscore_calibration_samples"]


_DEFAULT: Optional[Config] = None


def get_config() -> Config:
    global _DEFAULT
    if _DEFAULT is None:
        _DEFAULT = Config()
    return _DEFAULT
