"""End-to-end experiment drivers, one module per paper section.

Each pipeline wires the library components (``elicit``, ``judge``, ``prefill``,
``intervention``, ``capabilities``) into a runnable experiment that reads config,
writes JSONL artefacts under the run directory, and is resumable where the work
is expensive (rollouts, judging, continuations are append-and-skip).

The CLI (``gemma_distress.cli``) is a thin argparse dispatcher over these.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from ..config import Config
from ..io_utils import section_dir


def log(msg: str) -> None:
    """Single-line progress to stderr (stdout is reserved for any data output)."""
    print(f"[gemma-distress] {msg}", file=sys.stderr, flush=True)


def sampling(config: Config) -> dict[str, Any]:
    return config.experiment["sampling"]


def artefact(section: str, *parts: str) -> Path:
    """Path under ``runs/<section>/...`` (parent dirs are created)."""
    p = section_dir(section)
    for part in parts[:-1]:
        p = p / part
        p.mkdir(parents=True, exist_ok=True)
    return p / parts[-1] if parts else p
