"""Configuration loading.

A thin, attribute-accessible wrapper over the YAML config so call sites can
write ``cfg.welfare.monitor.threshold`` instead of nested dict indexing.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Iterable

import yaml

_DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "default.yaml"


class Config(dict):
    """Dict that also supports attribute access and nests recursively."""

    def __getattr__(self, name: str) -> Any:
        try:
            value = self[name]
        except KeyError as exc:  # pragma: no cover - defensive
            raise AttributeError(name) from exc
        return _wrap(value)

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value

    def get_path(self, dotted: str, default: Any = None) -> Any:
        node: Any = self
        for part in dotted.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return _wrap(node)


def _wrap(value: Any) -> Any:
    if isinstance(value, dict):
        return Config(value)
    if isinstance(value, list):
        return [_wrap(v) for v in value]
    return value


def load_config(path: str | os.PathLike | None = None,
                overrides: dict | None = None) -> Config:
    """Load YAML config, optionally deep-merging ``overrides`` on top."""
    cfg_path = Path(path) if path else _DEFAULT_CONFIG_PATH
    with open(cfg_path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if overrides:
        data = _deep_merge(data, overrides)
    return Config(data)


def _deep_merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for key, val in override.items():
        if key in out and isinstance(out[key], dict) and isinstance(val, dict):
            out[key] = _deep_merge(out[key], val)
        else:
            out[key] = val
    return out


def subject_by_key(cfg: Config, key: str) -> Config:
    for subj in cfg.subjects:
        if subj["key"] == key:
            return _wrap(subj)
    raise KeyError(f"No subject model with key {key!r} in config")


def condition_by_key(cfg: Config, key: str) -> Config:
    for cond in cfg.elicitation.conditions:
        if cond["key"] == key:
            return _wrap(cond)
    raise KeyError(f"No elicitation condition with key {key!r} in config")


def iter_subject_keys(cfg: Config) -> Iterable[str]:
    return (s["key"] for s in cfg.subjects)
