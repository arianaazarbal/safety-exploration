"""Configuration loading and lightweight typed accessors.

Config is a YAML file (see config/default.yaml). We keep it as nested dicts
plus a few small dataclasses for the objects that are passed around a lot
(models, generation settings). This avoids a heavy schema layer while still
giving call sites attribute access for the common cases.
"""
from __future__ import annotations

import copy
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "default.yaml"


@dataclass
class ModelSpec:
    """Static description of a target model.

    `role` is either "full" (open weights — can be evaluated, prefilled,
    finetuned and probed) or "eval" (closed weights — black-box evaluation
    only). Code that mutates or inspects internals must check this.
    """

    name: str
    backend: str            # hf | vllm | api
    family: str             # gemma | gemini
    role: str               # full | eval
    hf_id: str | None = None
    api_id: str | None = None
    api_provider: str | None = None
    chat_template: str = "auto"
    thinking: bool | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def is_open_weights(self) -> bool:
        return self.role == "full"

    def require_open_weights(self, what: str) -> None:
        if not self.is_open_weights:
            raise PermissionError(
                f"{what} requires open weights, but model '{self.name}' "
                f"(family={self.family}) is closed (role={self.role}). "
                "In this replication that operation is Gemma-only; see DESIGN.md."
            )


class Config:
    """Thin wrapper over the parsed YAML with helpers used across the package."""

    def __init__(self, data: dict[str, Any]):
        self._data = data

    # -- construction -------------------------------------------------------
    @classmethod
    def load(cls, path: str | os.PathLike | None = None) -> "Config":
        path = Path(path) if path else DEFAULT_CONFIG_PATH
        with open(path, "r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
        return cls(data)

    def copy(self) -> "Config":
        return Config(copy.deepcopy(self._data))

    # -- dict-style access --------------------------------------------------
    def __getitem__(self, key: str) -> Any:
        return self._data[key]

    def get(self, key: str, default: Any = None) -> Any:
        return self._data.get(key, default)

    @property
    def raw(self) -> dict[str, Any]:
        return self._data

    # -- model specs --------------------------------------------------------
    def model(self, name: str) -> ModelSpec:
        if name not in self._data["models"]:
            raise KeyError(
                f"Unknown model '{name}'. Known: {sorted(self._data['models'])}"
            )
        spec = dict(self._data["models"][name])
        known = {
            "backend", "family", "role", "hf_id", "api_id",
            "api_provider", "chat_template", "thinking",
        }
        extra = {k: v for k, v in spec.items() if k not in known}
        return ModelSpec(
            name=name,
            backend=spec["backend"],
            family=spec["family"],
            role=spec.get("role", "eval"),
            hf_id=spec.get("hf_id"),
            api_id=spec.get("api_id"),
            api_provider=spec.get("api_provider"),
            chat_template=spec.get("chat_template", "auto"),
            thinking=spec.get("thinking"),
            extra=extra,
        )

    @property
    def default_targets(self) -> list[str]:
        return list(self._data["default_targets"])

    # -- paths --------------------------------------------------------------
    def path(self, key: str) -> Path:
        p = Path(self._data["paths"][key])
        p.mkdir(parents=True, exist_ok=True)
        return p

    # -- welfare-scaled sample counts --------------------------------------
    def scale(self) -> float:
        return float(self._data.get("welfare", {}).get("scale", 1.0))

    def scaled_count(self, raw_count: int, minimum: int = 1) -> int:
        """Apply the welfare scale factor to a sample count.

        Always returns at least `minimum` so a scaled-down run still exercises
        every code path. See WELFARE.md for why the default footprint is small.
        """
        return max(minimum, round(raw_count * self.scale()))
