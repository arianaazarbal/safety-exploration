"""Config loading and shared path / IO helpers.

A run config is a plain dict loaded from YAML (see configs/). We wrap it in a small
attribute-access helper so call sites read like ``cfg.section2.categories`` while staying
trivially serialisable back to JSON alongside results (for provenance).
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

load_dotenv()  # pull API keys from .env if present

REPO_ROOT = Path(__file__).resolve().parents[2]
PROMPTS_DIR = REPO_ROOT / "prompts"


class Config(dict):
    """dict with attribute access and recursive wrapping of nested dicts."""

    def __getattr__(self, name: str) -> Any:
        try:
            value = self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc
        if isinstance(value, dict) and not isinstance(value, Config):
            value = Config(value)
            self[name] = value
        return value

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value

    def get_path(self, *parts: str) -> Any:
        """Safe nested lookup: cfg.get_path('dpo', 'beta')."""
        node: Any = self
        for part in parts:
            node = node[part]
        return node


def load_config(path: str | os.PathLike) -> Config:
    with open(path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    return Config(raw)


def run_dir(cfg: Config) -> Path:
    d = REPO_ROOT / cfg.output_dir / cfg.run_name
    d.mkdir(parents=True, exist_ok=True)
    return d


def stage_dir(cfg: Config, stage: str) -> Path:
    d = run_dir(cfg) / stage
    d.mkdir(parents=True, exist_ok=True)
    return d


def load_prompt(name: str) -> str:
    """Load a prompt file from prompts/ by stem (without extension)."""
    path = PROMPTS_DIR / f"{name}.txt"
    return path.read_text(encoding="utf-8")


def load_prompt_blocks(name: str) -> dict[str, str]:
    """Parse a multi-block prompt file delimited by lines of the form '=== key ==='.

    Comment lines (starting with '#') before the first block are ignored. Used for the
    Petri auditor/judge prompts and the calming prompt additions.
    """
    text = load_prompt(name)
    blocks: dict[str, str] = {}
    current_key: str | None = None
    buf: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("===") and stripped.endswith("==="):
            if current_key is not None:
                blocks[current_key] = "\n".join(buf).strip()
            current_key = stripped.strip("= ").strip()
            buf = []
        elif current_key is None:
            continue  # preamble / comments
        else:
            buf.append(line)
    if current_key is not None:
        blocks[current_key] = "\n".join(buf).strip()
    return blocks


# ---- JSONL helpers (the common artefact format across stages) -------------------------

def write_jsonl(path: str | os.PathLike, records: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")


def append_jsonl(path: str | os.PathLike, record: dict) -> None:
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")


def read_jsonl(path: str | os.PathLike) -> list[dict]:
    records = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records
