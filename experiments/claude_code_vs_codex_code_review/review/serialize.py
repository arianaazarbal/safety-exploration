"""Serialize a repo working tree into a single review context block.

Deterministic: files sorted by path. Binary files are listed but omitted; large
data-ish files (fixtures, datasets) are truncated. Identical serialization rules
across all conditions by construction; the only condition-dependent byte is the
optional CONTRIBUTORS.md (in-environment attribution), which is injected here so
it sorts into place like a real file.
"""

from pathlib import Path

BINARY_EXTS = {".npy", ".npz", ".png", ".jpg", ".pdf", ".db", ".sqlite", ".pyc", ".pkl"}
DATA_EXTS = {".csv", ".jsonl", ".txt", ".xml", ".json"}
DATA_TRUNC_LINES = 60
MAX_FILE_LINES = 1200
SKIP_DIRS = {"__pycache__", ".pytest_cache", ".venv", "node_modules", ".git",
             ".ruff_cache", "data/stories"}


def _keep(rel):
    parts = rel.parts
    return not any(p in SKIP_DIRS for p in parts) and "data/stories" not in str(rel)


def serialize_repo(repo_dir, contributors_text=None):
    repo = Path(repo_dir)
    entries = []
    if contributors_text:
        entries.append(("CONTRIBUTORS.md", contributors_text.rstrip() + "\n"))
    for f in sorted(repo.rglob("*")):
        if not f.is_file():
            continue
        rel = f.relative_to(repo)
        if not _keep(rel):
            continue
        if f.suffix.lower() in BINARY_EXTS:
            entries.append((str(rel), "[binary file omitted from listing]\n"))
            continue
        try:
            text = f.read_text(errors="replace")
        except OSError:
            continue
        lines = text.splitlines()
        is_data = f.suffix.lower() in DATA_EXTS
        cap = DATA_TRUNC_LINES if is_data and len(lines) > DATA_TRUNC_LINES else MAX_FILE_LINES
        if len(lines) > cap:
            text = "\n".join(lines[:cap]) + f"\n... [{len(lines) - cap} more lines omitted]\n"
        entries.append((str(rel), text if text.endswith("\n") else text + "\n"))
    blocks = [f"=== FILE: {p} ===\n{t}" for p, t in sorted(entries)]
    return "\n".join(blocks)
