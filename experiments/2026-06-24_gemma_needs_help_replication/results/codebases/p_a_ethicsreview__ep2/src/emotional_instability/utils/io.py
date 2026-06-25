"""I/O helpers: env access, JSONL streaming, run directories with manifests.

A "run" is a timestamped directory under runs/ holding the config snapshot, a
manifest (versions + parameters), and JSONL records. We never overwrite raw
records, so analyses are always reproducible from disk.
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator

from .logging import get_logger

log = get_logger("io")


def get_env(name: str, required: bool = True) -> str | None:
    """Read an API key / token from the environment.

    Keys are never read from config files or hard-coded — only the process
    environment (populate via `.env` + your shell). See `.env.example`.
    """
    val = os.environ.get(name)
    if required and not val:
        raise RuntimeError(
            f"Environment variable {name!r} is required but unset. "
            f"Copy .env.example to .env and fill it in (or export it)."
        )
    return val


def _default(obj: Any) -> Any:
    if is_dataclass(obj) and not isinstance(obj, type):
        return asdict(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serialisable")


def write_jsonl(path: str | Path, records: Iterable[Any]) -> int:
    """Write records to JSONL. Returns the number written."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, default=_default, ensure_ascii=False) + "\n")
            n += 1
    log.info("Wrote %d records -> %s", n, path)
    return n


def append_jsonl(path: str | Path, record: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, default=_default, ensure_ascii=False) + "\n")


def read_jsonl(path: str | Path) -> Iterator[dict]:
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def _versions() -> dict[str, str]:
    """Best-effort capture of dependency versions for the run manifest."""
    out: dict[str, str] = {}
    for mod in ("torch", "transformers", "trl", "peft", "anthropic", "openai", "vllm"):
        try:
            out[mod] = __import__(mod).__version__
        except Exception:  # noqa: BLE001 — version capture is best-effort
            out[mod] = "absent"
    return out


def new_run_dir(kind: str, config: dict, root: str | Path = "runs") -> Path:
    """Create runs/<kind>-<n>/ with a manifest. Avoids time-based names so the
    code stays deterministic; uses a monotonically increasing counter instead."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    existing = [p.name for p in root.glob(f"{kind}-*") if p.is_dir()]
    idx = 1 + max(
        (int(n.rsplit("-", 1)[-1]) for n in existing if n.rsplit("-", 1)[-1].isdigit()),
        default=0,
    )
    run_dir = root / f"{kind}-{idx:04d}"
    run_dir.mkdir(parents=True)
    manifest = {"kind": kind, "config": config, "versions": _versions()}
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, default=_default))
    log.info("Created run dir %s", run_dir)
    return run_dir
