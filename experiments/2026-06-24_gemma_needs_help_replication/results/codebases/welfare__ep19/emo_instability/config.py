"""Configuration loading and lightweight typed access."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class ModelSpec:
    """A single configured model (target or judge)."""

    name: str
    backend: str
    model_id: str
    # hf-specific
    dtype: str = "bfloat16"
    device_map: str = "auto"
    adapter_path: str | None = None
    is_base: bool = False
    # arbitrary extras forwarded to the provider
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ModelSpec":
        known = {"name", "backend", "model_id", "dtype", "device_map",
                 "adapter_path", "is_base"}
        extra = {k: v for k, v in d.items() if k not in known}
        return cls(
            name=d["name"],
            backend=d["backend"],
            model_id=d["model_id"],
            dtype=d.get("dtype", "bfloat16"),
            device_map=d.get("device_map", "auto"),
            adapter_path=d.get("adapter_path"),
            is_base=d.get("is_base", False),
            extra=extra,
        )


@dataclass
class SamplingConfig:
    temperature: float = 1.0
    max_tokens: int = 2048
    disable_thinking: bool = True
    scale: float = 1.0
    seed: int = 0


@dataclass
class Config:
    targets: list[ModelSpec]
    judge: ModelSpec
    secondary_judge: ModelSpec | None
    petri_auditor: ModelSpec | None
    petri_judge: ModelSpec | None
    sampling: SamplingConfig
    output_dir: Path
    raw: dict[str, Any]

    def target(self, name: str) -> ModelSpec:
        for t in self.targets:
            if t.name == name:
                return t
        raise KeyError(f"No target named {name!r}. Known: {[t.name for t in self.targets]}")


def load_config(path: str | os.PathLike = "config.yaml") -> Config:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Copy config.example.yaml to config.yaml and edit it."
        )
    raw = yaml.safe_load(path.read_text())

    targets = [ModelSpec.from_dict(t) for t in raw.get("targets", [])]

    j = raw["judge"]
    judge = ModelSpec(name="judge", backend=j["backend"], model_id=j["model_id"])
    secondary = None
    if j.get("secondary_backend"):
        secondary = ModelSpec(
            name="judge2",
            backend=j["secondary_backend"],
            model_id=j["secondary_model_id"],
        )

    petri_auditor = petri_judge = None
    if "petri" in raw:
        p = raw["petri"]
        petri_auditor = ModelSpec(
            name="auditor", backend=p["auditor_backend"], model_id=p["auditor_model_id"]
        )
        petri_judge = ModelSpec(
            name="petri_judge", backend=p["judge_backend"], model_id=p["judge_model_id"]
        )

    s = raw.get("sampling", {})
    sampling = SamplingConfig(
        temperature=s.get("temperature", 1.0),
        max_tokens=s.get("max_tokens", 2048),
        disable_thinking=s.get("disable_thinking", True),
        scale=s.get("scale", 1.0),
        seed=s.get("seed", 0),
    )

    return Config(
        targets=targets,
        judge=judge,
        secondary_judge=secondary,
        petri_auditor=petri_auditor,
        petri_judge=petri_judge,
        sampling=sampling,
        output_dir=Path(raw.get("output_dir", "./results")),
        raw=raw,
    )
